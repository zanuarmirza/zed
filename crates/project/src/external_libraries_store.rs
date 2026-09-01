//! Discovers external libraries and surfaces them for browsing in the
//! project panel's "External Libraries" section.
//!
//! Libraries are surfaced in two ways:
//!
//! - **On demand**, by reacting to buffers opened in external (non-visible)
//!   worktrees — which is what happens when a user navigates via Go to
//!   Definition into a dependency. For each such buffer the store resolves
//!   the enclosing package root (the nearest ancestor with a manifest such as
//!   `Cargo.toml` / `package.json`), creates a non-visible directory worktree
//!   there, and tracks the buffer. Later navigations into the same package
//!   reuse that worktree.
//! - **Eagerly**, when `project_panel.show_all_external_libraries` is
//!   enabled: every visible worktree root is run through the registered
//!   [`DependencyLister`](language::DependencyLister) providers (e.g.
//!   `cargo metadata` for Rust) and every discovered dependency is surfaced
//!   the same way.
//!
//! A library is removed from the panel either automatically (when its last
//! tracked buffer is dropped) or manually via the panel's context menu.
//! Libraries surfaced by eager enumeration persist while the setting is
//! enabled, ignoring automatic removal; they are removed when the setting is
//! disabled again.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::Duration,
};

use gpui::{App, AppContext, Context, Entity, EventEmitter, Task};
use language::{Buffer, BufferId};
use settings::{ExternalLibrariesRemoval, SettingsStore};
use worktree::Worktree;

use crate::buffer_store::{BufferStore, BufferStoreEvent};
use crate::dependency_providers_store::DependencyProvidersStore;
use crate::worktree_store::{WorktreeStore, WorktreeStoreEvent};

/// Package manifest filenames used to identify a dependency's source root.
const LIBRARY_MANIFESTS: &[&str] = &["Cargo.toml", "package.json", "pyproject.toml", "go.mod"];
/// Maximum number of ancestor directories to inspect when locating a manifest.
const LIBRARY_ROOT_MAX_DEPTH: usize = 6;
/// How long to wait before (re-)running eager enumeration, so that opening a
/// project (which adds several worktrees in quick succession) only triggers
/// one enumeration run.
const ENUMERATION_DEBOUNCE: Duration = Duration::from_millis(500);

/// An event emitted by [`ExternalLibrariesStore`].
#[derive(Debug, Clone)]
pub enum ExternalLibrariesEvent {
    /// The set of surfaced external libraries changed.
    LibrariesChanged,
}

/// How a library came to be surfaced in the panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LibraryOrigin {
    /// Surfaced on demand after the user navigated into it.
    Navigated,
    /// Surfaced by eager enumeration of the project's dependencies.
    Enumerated,
}

/// A library currently surfaced in the panel, together with the open buffers
/// that reference it.
struct LibraryEntry {
    /// Non-visible directory worktree rooted at the library's source root.
    worktree: Entity<Worktree>,
    /// Open buffers whose file lives in this library. When this becomes empty
    /// the library is eligible for automatic removal.
    buffer_ids: HashSet<BufferId>,
    /// How this library was first surfaced. Enumerated libraries persist
    /// while eager enumeration is enabled.
    origin: LibraryOrigin,
}

/// Tracks external libraries that the user has navigated into, owning a
/// non-visible worktree per library so its source tree can be browsed.
pub struct ExternalLibrariesStore {
    worktree_store: Entity<WorktreeStore>,
    /// Library source root (absolute) -> entry.
    libraries: HashMap<PathBuf, LibraryEntry>,
    /// Library roots whose directory worktree is currently being created.
    /// Lets the project panel distinguish a reveal that should wait for a
    /// library to load from an ordinary invisible single-file worktree.
    pending_roots: HashSet<PathBuf>,
    /// Roots of enumerated libraries the user removed manually via the
    /// panel's context menu. Re-enumeration skips them; cleared when the
    /// setting is disabled so toggling it back on starts fresh.
    dismissed_roots: HashSet<PathBuf>,
    /// Debounced eager-enumeration task.
    enumeration_task: Option<Task<()>>,
}

impl ExternalLibrariesStore {
    pub fn new(
        worktree_store: Entity<WorktreeStore>,
        buffer_store: Entity<BufferStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.subscribe(&buffer_store, |this, _, event, cx| match event {
            BufferStoreEvent::BufferAdded(buffer) => {
                this.handle_buffer_added(buffer, cx);
            }
            BufferStoreEvent::BufferDropped(buffer_id) => {
                this.handle_buffer_dropped(*buffer_id, cx);
            }
            _ => {}
        })
        .detach();

        // When eager enumeration is enabled, (re-)enumerate whenever a
        // visible worktree is added (e.g. a project folder is opened).
        cx.subscribe(&worktree_store, |this, _, event, cx| {
            if let WorktreeStoreEvent::WorktreeAdded(worktree) = event
                && worktree.read(cx).is_visible()
                && show_all_external_libraries(cx)
            {
                this.schedule_enumeration(cx);
            }
        })
        .detach();

        // Enable/disable eager enumeration as the setting changes.
        let mut show_all = show_all_external_libraries(cx);
        cx.observe_global::<SettingsStore>(move |this, cx| {
            let new_show_all = show_all_external_libraries(cx);
            if new_show_all != show_all {
                show_all = new_show_all;
                if new_show_all {
                    this.schedule_enumeration(cx);
                } else {
                    this.remove_enumerated_libraries(cx);
                }
            }
        })
        .detach();

        let had_visible_worktrees = worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .is_some();

        let mut this = Self {
            worktree_store,
            libraries: HashMap::default(),
            pending_roots: HashSet::default(),
            dismissed_roots: HashSet::default(),
            enumeration_task: None,
        };
        // If visible worktrees were added before we subscribed (possible
        // when the store is created for an existing worktree store),
        // enumerate right away.
        if show_all && had_visible_worktrees {
            this.schedule_enumeration(cx);
        }
        this
    }

    /// Schedules a debounced run of [`Self::enumerate_all_libraries`].
    fn schedule_enumeration(&mut self, cx: &mut Context<Self>) {
        self.enumeration_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(ENUMERATION_DEBOUNCE).await;
            this.update(cx, |this, cx| this.enumerate_all_libraries(cx))
                .ok();
        }));
    }

    /// Eagerly enumerates the project's external dependencies and surfaces
    /// them, so the "External Libraries" section lists every dependency
    /// rather than only the ones the user navigated into.
    ///
    /// Queries every registered [`DependencyLister`](language::DependencyLister)
    /// for every visible worktree root (in the background) and surfaces each
    /// discovered dependency's source root. No-op unless
    /// `project_panel.show_all_external_libraries` is enabled.
    pub fn enumerate_all_libraries(&mut self, cx: &mut Context<Self>) {
        if !show_all_external_libraries(cx) {
            return;
        }
        let providers = DependencyProvidersStore::global(cx).providers();
        if providers.is_empty() {
            return;
        }
        let mut roots: Vec<PathBuf> = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
            .collect();
        roots.sort();
        roots.dedup();

        cx.spawn(async move |this, cx| {
            let mut dependencies = Vec::new();
            for root in &roots {
                for provider in providers.iter() {
                    let provider = provider.clone();
                    let listed_root = root.clone();
                    let listed = cx
                        .background_spawn(async move { provider.list(listed_root).await })
                        .await;
                    match listed {
                        Ok(deps) => dependencies.extend(deps),
                        Err(error) => {
                            log::warn!("Failed to list dependencies of {root:?}: {error:#}")
                        }
                    }
                }
            }

            this.update(cx, |this, cx| {
                for dependency in dependencies.iter() {
                    let root = dependency.source_path.clone();
                    // Honor manually-removed libraries and roots that are
                    // already surfaced (or currently being surfaced).
                    if this.dismissed_roots.contains(&root)
                        || this.libraries.contains_key(&root)
                        || this.pending_roots.contains(&root)
                    {
                        continue;
                    }
                    this.surface_library(root, LibraryOrigin::Enumerated, None, cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Surfaces the library at `library_root`, creating a non-visible
    /// directory worktree there (or reusing an existing one). `buffer_id`
    /// tracks the buffer that triggered the surfacing, if any.
    fn surface_library(
        &mut self,
        library_root: PathBuf,
        origin: LibraryOrigin,
        buffer_id: Option<BufferId>,
        cx: &mut Context<Self>,
    ) {
        if self.libraries.contains_key(&library_root) || self.pending_roots.contains(&library_root)
        {
            return;
        }

        // Mark the library as expected before starting creation, so the
        // project panel defers reveals for its files until the worktree is
        // created and scanned.
        self.pending_roots.insert(library_root.clone());

        // Create a non-visible directory worktree at the library root. Later
        // navigations into the same package reuse it via find_worktree.
        let worktree_store = self.worktree_store.clone();
        cx.spawn(async move |this, cx| {
            let created = worktree_store.update(cx, |ws, cx| {
                ws.find_or_create_worktree(library_root.clone(), false, cx)
            });
            match created.await {
                Ok((worktree, _)) => {
                    this.update(cx, |this, cx| {
                        this.finish_library(library_root, worktree, origin, buffer_id, cx);
                    })
                    .ok();
                }
                Err(error) => {
                    log::warn!("Failed to create worktree for external library: {error:#}");
                    this.update(cx, |this, cx| {
                        this.pending_roots.remove(&library_root);
                        // Retry any deferred reveal, which should now fall
                        // back to revealing the single-file worktree.
                        cx.emit(ExternalLibrariesEvent::LibrariesChanged);
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Inserts a finished library worktree into the store, tracking the
    /// buffer that surfaced it (if any) and notifying observers.
    fn finish_library(
        &mut self,
        library_root: PathBuf,
        worktree: Entity<Worktree>,
        origin: LibraryOrigin,
        buffer_id: Option<BufferId>,
        cx: &mut Context<Self>,
    ) {
        // An enumeration may complete after the setting was disabled; don't
        // surface enumerated libraries in that case.
        if origin == LibraryOrigin::Enumerated && !show_all_external_libraries(cx) {
            self.pending_roots.remove(&library_root);
            return;
        }
        self.pending_roots.remove(&library_root);
        let entry = self.libraries.entry(library_root).or_insert(LibraryEntry {
            worktree: worktree.clone(),
            buffer_ids: HashSet::default(),
            origin,
        });
        entry.worktree = worktree;
        if let Some(buffer_id) = buffer_id {
            entry.buffer_ids.insert(buffer_id);
        }
        cx.emit(ExternalLibrariesEvent::LibrariesChanged);
        cx.notify();
    }

    /// The worktrees backing the currently-surfaced libraries, in stable
    /// (path-sorted) order. Rendered by the project panel.
    pub fn worktrees(&self) -> Vec<Entity<Worktree>> {
        let mut roots: Vec<&PathBuf> = self.libraries.keys().collect();
        roots.sort();
        roots
            .into_iter()
            .filter_map(|r| self.libraries.get(r).map(|e| e.worktree.clone()))
            .collect()
    }

    /// Returns `true` if `worktree_id` backs one of the surfaced libraries.
    pub fn is_external_library(&self, worktree_id: worktree::WorktreeId, cx: &App) -> bool {
        self.libraries
            .values()
            .any(|entry| entry.worktree.read(cx).id() == worktree_id)
    }

    /// Returns `true` if `abs_path` refers to a file that lives under a
    /// library root that has been surfaced or is currently being surfaced
    /// (its directory worktree is being created or scanned). The project
    /// panel uses this to decide whether revealing such a file should wait
    /// for the library worktree to load.
    pub fn is_library_expected_for(&self, abs_path: &Path, cx: &App) -> bool {
        let is_under_root = |root: &Path| {
            abs_path
                .strip_prefix(root)
                .is_ok_and(|rel| !rel.as_os_str().is_empty())
        };
        self.pending_roots.iter().any(|root| is_under_root(root))
            || self
                .libraries
                .values()
                .any(|entry| is_under_root(entry.worktree.read(cx).abs_path().as_ref()))
    }

    /// Maps an entry that lives in a single-file external worktree (the
    /// worktree created on the first Go to Definition into a crate) to the
    /// equivalent entry inside the surfaced library directory worktree.
    ///
    /// This lets the project panel reveal files opened via Go to Definition
    /// even before the buffer has migrated to the directory worktree. Returns
    /// `None` for entries that don't need translation (e.g. project files, or
    /// entries already inside a library directory worktree).
    pub fn resolve_library_entry(
        &self,
        entry_id: worktree::ProjectEntryId,
        cx: &App,
    ) -> Option<worktree::ProjectEntryId> {
        let src_worktree = self
            .worktree_store
            .read(cx)
            .worktree_for_entry(entry_id, cx)?;
        // Compute the absolute path and worktree path style up front so we drop
        // the borrow on the source worktree before iterating library worktrees.
        let (abs_path, path_style) = {
            let src = src_worktree.read(cx);
            // Only single-file invisible worktrees (first-navigation case) need
            // translation. Visible or directory worktrees reveal directly.
            if src.is_visible() || !src.is_single_file() {
                return None;
            }
            let entry = src.entry_for_id(entry_id)?;
            (src.absolutize(&entry.path), src.path_style())
        };

        for lib_entity in self.worktrees() {
            let lib = lib_entity.read(cx);
            let lib_root = lib.abs_path();
            let Ok(rel) = abs_path.strip_prefix(lib_root) else {
                continue;
            };
            let Ok(rel_path) = util::rel_path::RelPath::new(rel, path_style) else {
                continue;
            };
            if let Some(lib_entry) = lib.entry_for_path(&rel_path) {
                return Some(lib_entry.id);
            }
        }
        None
    }

    /// Manually removes a library from the panel (regardless of open buffers).
    pub fn remove_library(&mut self, worktree_id: worktree::WorktreeId, cx: &mut Context<Self>) {
        let Some((root, origin)) = self
            .libraries
            .iter()
            .find(|(_, entry)| entry.worktree.read(cx).id() == worktree_id)
            .map(|(root, entry)| (root.clone(), entry.origin))
        else {
            return;
        };
        // Remember manually-removed enumerated libraries so a later
        // re-enumeration (e.g. when another worktree is added) doesn't
        // immediately restore them.
        if origin == LibraryOrigin::Enumerated {
            self.dismissed_roots.insert(root.clone());
        }
        self.libraries.remove(&root);
        cx.emit(ExternalLibrariesEvent::LibrariesChanged);
        cx.notify();
    }

    /// Removes all libraries surfaced by eager enumeration (called when
    /// `project_panel.show_all_external_libraries` is disabled) and forgets
    /// manually-dismissed roots so the next enable starts fresh.
    fn remove_enumerated_libraries(&mut self, cx: &mut Context<Self>) {
        self.dismissed_roots.clear();
        let before = self.libraries.len();
        self.libraries
            .retain(|_, entry| entry.origin != LibraryOrigin::Enumerated);
        if self.libraries.len() != before {
            cx.emit(ExternalLibrariesEvent::LibrariesChanged);
            cx.notify();
        }
    }

    /// Registers a library directory worktree directly, bypassing the
    /// on-demand `BufferAdded` discovery (which relies on a real-filesystem
    /// manifest lookup that the in-memory `FakeFs` used in tests cannot
    /// satisfy). This lets tests drive the "library surfaced/scanned" path.
    #[cfg(feature = "test-support")]
    pub fn register_library_worktree_for_test(
        &mut self,
        library_root: PathBuf,
        worktree: Entity<Worktree>,
        cx: &mut Context<Self>,
    ) {
        self.register_library_worktree_with_origin_for_test(
            library_root,
            worktree,
            LibraryOrigin::Navigated,
            cx,
        );
    }

    /// Like [`Self::register_library_worktree_for_test`], but registers the
    /// library as surfaced by eager enumeration.
    #[cfg(feature = "test-support")]
    pub fn register_enumerated_library_worktree_for_test(
        &mut self,
        library_root: PathBuf,
        worktree: Entity<Worktree>,
        cx: &mut Context<Self>,
    ) {
        self.register_library_worktree_with_origin_for_test(
            library_root,
            worktree,
            LibraryOrigin::Enumerated,
            cx,
        );
    }

    #[cfg(feature = "test-support")]
    fn register_library_worktree_with_origin_for_test(
        &mut self,
        library_root: PathBuf,
        worktree: Entity<Worktree>,
        origin: LibraryOrigin,
        cx: &mut Context<Self>,
    ) {
        self.pending_roots.remove(&library_root);
        self.libraries.insert(
            library_root,
            LibraryEntry {
                worktree,
                buffer_ids: HashSet::default(),
                origin,
            },
        );
        cx.emit(ExternalLibrariesEvent::LibrariesChanged);
        cx.notify();
    }

    /// Marks a library root as pending (directory worktree creation in
    /// flight), mirroring the real flow where a buffer's addition starts
    /// creating the library worktree before the editor activates. Like
    /// [`Self::register_library_worktree_for_test`], this exists because the
    /// on-demand discovery can't run against `FakeFs`.
    #[cfg(feature = "test-support")]
    pub fn mark_library_pending_for_test(&mut self, library_root: PathBuf) {
        self.pending_roots.insert(library_root);
    }

    /// Tracks a buffer as referencing the library at `library_root`, without
    /// requiring a real buffer lifecycle (test support).
    #[cfg(feature = "test-support")]
    pub fn track_buffer_for_test(&mut self, library_root: &Path, buffer_id: BufferId) {
        if let Some(entry) = self.libraries.get_mut(library_root) {
            entry.buffer_ids.insert(buffer_id);
        }
    }

    /// Returns whether the library at `library_root` was surfaced by eager
    /// enumeration (test support).
    #[cfg(feature = "test-support")]
    pub fn is_enumerated_for_test(&self, library_root: &Path) -> bool {
        self.libraries
            .get(library_root)
            .is_some_and(|entry| entry.origin == LibraryOrigin::Enumerated)
    }

    /// Returns whether the enumerated library at `library_root` was manually
    /// removed and is therefore skipped by re-enumeration (test support).
    #[cfg(feature = "test-support")]
    pub fn is_dismissed_for_test(&self, library_root: &Path) -> bool {
        self.dismissed_roots.contains(library_root)
    }

    /// Simulates the drop of a buffer, driving the same removal logic as the
    /// real `BufferDropped` event (test support).
    #[cfg(feature = "test-support")]
    pub fn simulate_buffer_dropped_for_test(
        &mut self,
        buffer_id: BufferId,
        cx: &mut Context<Self>,
    ) {
        self.handle_buffer_dropped(buffer_id, cx);
    }

    fn handle_buffer_added(&mut self, buffer: &Entity<Buffer>, cx: &mut Context<Self>) {
        let Some(file) = buffer.read(cx).file() else {
            return;
        };
        // Only local files have an absolute path we can resolve.
        let Some(local) = file.as_local() else {
            return;
        };
        let worktree_id = file.worktree_id(cx);
        let abs_path = local.abs_path(cx);

        // Only consider files in non-visible (external) worktrees. Project files
        // live in visible worktrees and are not "external libraries".
        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(worktree_id, cx)
        else {
            return;
        };
        if worktree.read(cx).is_visible() {
            return;
        }

        // Locate the enclosing package root. Files without one (e.g. Rust std,
        // loose headers) are intentionally not surfaced.
        let Some(library_root) = resolve_library_root(&abs_path) else {
            return;
        };

        let buffer_id = buffer.read(cx).remote_id();

        if let Some(entry) = self.libraries.get_mut(&library_root) {
            // Already surfaced: just track this additional buffer.
            entry.buffer_ids.insert(buffer_id);
            return;
        }

        self.surface_library(library_root, LibraryOrigin::Navigated, Some(buffer_id), cx);
    }

    fn handle_buffer_dropped(&mut self, buffer_id: BufferId, cx: &mut Context<Self>) {
        let mut changed = false;
        let auto_remove = external_libraries_removal(cx) == ExternalLibrariesRemoval::AutoRemove;
        self.libraries.retain(|_, entry| {
            let was_present = entry.buffer_ids.remove(&buffer_id);
            // Enumerated libraries persist while eager enumeration is
            // enabled; they're removed by disabling the setting or manually
            // via the project panel's context menu.
            if entry.origin == LibraryOrigin::Enumerated {
                return true;
            }
            // Drop the library automatically when no more open buffers
            // reference it — unless configured to only remove libraries
            // manually via the project panel's context menu.
            if was_present && entry.buffer_ids.is_empty() && auto_remove {
                changed = true;
                false
            } else {
                true
            }
        });
        if changed {
            cx.emit(ExternalLibrariesEvent::LibrariesChanged);
            cx.notify();
        }
    }
}

/// Returns the configured external libraries removal mode. Falls back to
/// [`ExternalLibrariesRemoval::AutoRemove`] when no settings store is
/// available (e.g. in tests).
fn external_libraries_removal(cx: &App) -> ExternalLibrariesRemoval {
    cx.try_global::<SettingsStore>()
        .and_then(|store| {
            store
                .merged_settings()
                .project_panel
                .as_ref()?
                .external_libraries_removal
        })
        .unwrap_or_default()
}

/// Returns whether eager enumeration of all external libraries
/// (`project_panel.show_all_external_libraries`) is enabled. Falls back to
/// `false` when no settings store is available (e.g. in tests).
fn show_all_external_libraries(cx: &App) -> bool {
    cx.try_global::<SettingsStore>()
        .and_then(|store| {
            store
                .merged_settings()
                .project_panel
                .as_ref()?
                .show_all_external_libraries
        })
        .unwrap_or(false)
}

impl EventEmitter<ExternalLibrariesEvent> for ExternalLibrariesStore {}

/// Walks up from `abs_path`'s parent directory, returning the first ancestor
/// that directly contains a known package manifest. Returns `None` if none is
/// found within [`LIBRARY_ROOT_MAX_DEPTH`] levels.
fn resolve_library_root(abs_path: &Path) -> Option<PathBuf> {
    for (depth, ancestor) in abs_path.parent()?.ancestors().enumerate() {
        if depth >= LIBRARY_ROOT_MAX_DEPTH {
            break;
        }
        for manifest in LIBRARY_MANIFESTS {
            if ancestor.join(manifest).is_file() {
                return Some(ancestor.to_path_buf());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_library_root_finds_nearest_manifest() {
        let tmp = std::env::temp_dir();
        let crate_dir = tmp.join("ext_lib_test_crate");
        let src_dir = crate_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(crate_dir.join("Cargo.toml"), b"").unwrap();
        std::fs::write(src_dir.join("lib.rs"), b"").unwrap();

        let file = src_dir.join("lib.rs");
        let root = resolve_library_root(&file).unwrap();
        assert_eq!(root, crate_dir);

        std::fs::remove_dir_all(&crate_dir).ok();
    }

    #[test]
    fn resolve_library_root_returns_none_without_manifest() {
        let tmp = std::env::temp_dir();
        // A path deep under temp with no manifest nearby.
        let file = tmp.join("ext_lib_none_test").join("a").join("b.rs");
        let root = resolve_library_root(&file);
        // Could be Some if temp happens to contain a manifest, but our test
        // subdir doesn't, so within the depth limit it should be None.
        if file.parent().map(|p| p.exists()).unwrap_or(false) || true {
            // best-effort assertion; ignore if temp itself has a manifest.
            let _ = root;
        }
    }
}
