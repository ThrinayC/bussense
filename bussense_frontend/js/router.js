/**
 * BusSense SPA Router
 * Hash-based client-side routing with page lazy loading.
 */
import { ICONS } from './icons.js';

const routes = {};
let currentPage = null;

export function registerRoute(hash, loader) {
  routes[hash] = loader;
}

export async function navigate(hash) {
  const target = hash || '#overview';
  const loader = routes[target];

  // Update sidebar active state
  document.querySelectorAll('.sidebar-link').forEach(el => {
    el.classList.toggle('active', el.getAttribute('data-route') === target);
  });

  const container = document.getElementById('page-container');
  if (!container) return;

  if (!loader) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">${ICONS.not_found}</div><h3>Page not found</h3><p>The page "${target}" does not exist.</p></div>`;
    return;
  }

  // Show loading
  container.innerHTML = `
    <div style="padding: 48px; display: flex; align-items: center; justify-content: center;">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 12px;">
        <div class="loading-shimmer" style="width: 200px; height: 20px;"></div>
        <div class="loading-shimmer" style="width: 160px; height: 14px;"></div>
      </div>
    </div>`;

  try {
    const pageModule = await loader();
    if (currentPage && currentPage.destroy) {
      currentPage.destroy();
    }
    container.innerHTML = '';
    currentPage = pageModule;
    await pageModule.render(container);
    // Smooth entrance
    container.classList.remove('page-fade-in');
    void container.offsetWidth; // force reflow
    container.classList.add('page-fade-in');
  } catch (err) {
    console.error(`Failed to load page ${target}:`, err);
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">${ICONS.error_page}</div><h3>Error loading page</h3><p>${err.message}</p></div>`;
  }
}

export function initRouter() {
  window.addEventListener('hashchange', () => navigate(location.hash));
  // Initial navigation
  navigate(location.hash || '#overview');
}
