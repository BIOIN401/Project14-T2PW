/* Site chrome: mobile nav toggle + About dropdown (click-outside / Escape). */
(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  ready(function () {
    document.documentElement.classList.add("js");

    var header = document.querySelector(".site-header");
    if (!header) {
      return;
    }

    var toggle = header.querySelector("[data-nav-toggle]");
    var nav = header.querySelector(".site-nav");
    var about = header.querySelector(".nav-dropdown details");

    function setNavOpen(open) {
      if (!nav || !toggle) {
        return;
      }
      nav.classList.toggle("is-open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (!open) {
        toggle.focus();
      }
    }

    function closeAbout() {
      if (about && about.open) {
        about.open = false;
      }
    }

    function closeAll() {
      setNavOpen(false);
      closeAbout();
    }

    if (toggle && nav) {
      toggle.addEventListener("click", function () {
        var open = !nav.classList.contains("is-open");
        if (open) {
          closeAbout();
        }
        setNavOpen(open);
      });
    }

    if (about) {
      about.addEventListener("toggle", function () {
        if (about.open && nav && nav.classList.contains("is-open") === false) {
          /* desktop: leave mobile alone */
        }
        if (about.open && toggle && window.matchMedia("(max-width: 48rem)").matches) {
          /* keep mobile panel open while using About */
          setNavOpen(true);
        }
      });
    }

    document.addEventListener("click", function (ev) {
      if (!header.contains(ev.target)) {
        closeAll();
        return;
      }
      if (about && about.open && !about.contains(ev.target)) {
        closeAbout();
      }
    });

    document.addEventListener("keydown", function (ev) {
      if (ev.key !== "Escape") {
        return;
      }
      if (about && about.open) {
        closeAbout();
        var summary = about.querySelector("summary");
        if (summary) {
          summary.focus();
        }
        ev.preventDefault();
        return;
      }
      if (nav && nav.classList.contains("is-open")) {
        setNavOpen(false);
        ev.preventDefault();
      }
    });

    if (nav) {
      nav.addEventListener("click", function (ev) {
        var link = ev.target.closest("a");
        if (!link || !nav.contains(link)) {
          return;
        }
        /* Follow the link; close the mobile panel afterward for back-navigation cleanliness. */
        if (window.matchMedia("(max-width: 48rem)").matches) {
          setNavOpen(false);
        }
      });
    }
  });
})();
