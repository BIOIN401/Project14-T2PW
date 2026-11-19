/* Submit form: reveal optional pathway rows one at a time (max from data-max-pathways). */
(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function clearInputs(root) {
    root.querySelectorAll("input, textarea, select").forEach(function (el) {
      if (el.type === "radio" || el.type === "checkbox" || el.type === "hidden") {
        return;
      }
      if (el.type === "file") {
        el.value = "";
        return;
      }
      el.value = "";
    });
    root.querySelectorAll(".field__error").forEach(function (el) {
      el.remove();
    });
  }

  function visibleRows(rows) {
    return Array.prototype.filter.call(rows, function (row) {
      return !row.hasAttribute("hidden");
    });
  }

  function syncAddButton(form) {
    var add = form.querySelector("[data-add-pathway]");
    if (!add) {
      return;
    }
    var next = form.querySelector(".pathway-row[hidden]");
    add.hidden = !next;
    add.disabled = !next;
  }

  ready(function () {
    var form = document.querySelector("form.submit");
    if (!form) {
      return;
    }

    form.addEventListener("click", function (ev) {
      var add = ev.target.closest("[data-add-pathway]");
      if (add && form.contains(add)) {
        ev.preventDefault();
        var next = form.querySelector(".pathway-row[hidden]");
        if (!next) {
          return;
        }
        next.removeAttribute("hidden");
        var focus = next.querySelector("input, textarea, select");
        if (focus) {
          focus.focus();
        }
        syncAddButton(form);
        return;
      }

      var remove = ev.target.closest("[data-remove-pathway]");
      if (remove && form.contains(remove)) {
        ev.preventDefault();
        var row = remove.closest(".pathway-row");
        if (!row || row.getAttribute("data-pathway") === "1") {
          return;
        }
        clearInputs(row);
        row.setAttribute("hidden", "");
        // Keep numbering stable: if a higher row is still visible, leave it;
        // only re-hide empty trailing gaps after the last filled visible row.
        syncAddButton(form);
      }
    });

    syncAddButton(form);
  });
})();
