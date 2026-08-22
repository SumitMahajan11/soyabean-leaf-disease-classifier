
/*
 * Soybean Disease Detection System
 * Frontend Utility Scripts - v2.2.0
 */

// Mobile menu functionality
function toggleMobileMenu() {
  const navMenu = document.querySelector('.nav-menu');
  const hamburger = document.querySelector('.hamburger');
  if (navMenu) {
    navMenu.classList.toggle('show');
    if (hamburger) hamburger.classList.toggle('active');
    console.log('📱 Mobile menu toggled');
  }
}

function closeMobileMenu() {
  const navMenu = document.querySelector('.nav-menu');
  const hamburger = document.querySelector('.hamburger');
  if (navMenu) {
    navMenu.classList.remove('show');
    if (hamburger) hamburger.classList.remove('active');
  }
}

// Initialize hamburger menu for mobile views
function initHamburgerMenu() {
  // Check if hamburger already exists
  if (document.querySelector('.hamburger')) return;
  
  const hamburger = document.createElement('div');
  hamburger.className = 'hamburger';
  hamburger.innerHTML = '<span></span><span></span><span></span>';
  hamburger.title = 'Toggle Menu';
  hamburger.addEventListener('click', toggleMobileMenu);
  
  const header = document.querySelector('.header-content');
  if (header) {
    const navMenu = header.querySelector('.nav-menu');
    if (navMenu) {
      navMenu.insertAdjacentElement('beforebegin', hamburger);
      console.log('✅ Hamburger menu initialized');
    }
  }
}

// Auto-close menu when clicking links
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('nav-link')) {
    closeMobileMenu();
  }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
  initHamburgerMenu();
  
  // Initialize Intersection Observer for animations if not in index.html
  if (typeof initIntersectionObserver === 'function') {
    initIntersectionObserver();
  }
});

/**
 * Utility: Scroll to section
 */
function scrollToSection(id) {
  const section = document.getElementById(id);
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
    closeMobileMenu();
  }
}
