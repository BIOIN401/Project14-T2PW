/* Submit form: reveal optional pathway rows; remove compact-renumbers remaining rows. */
(function () {
  var FIELD_BASES = ["name", "organism", "pmids", "pdfs"];

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

  function syncAddButton(form) {
    var add = form.querySelector("[data-add-pathway]");
    if (!add) {
      return;
    }
    var next = form.querySelector(".pathway-row[hidden]");
    add.hidden = !next;
    add.disabled = !next;
  }

  function rewriteAttr(el, attr, fromNum, toNum) {
    var val = el.getAttribute(attr);
    if (!val) {
      return;
    }
    var next = val;
    FIELD_BASES.forEach(function (base) {
      var re = new RegExp("\\b" + base + "_" + fromNum + "\\b", "g");
      next = next.replace(re, base + "_" + toNum);
    });
    if (next !== val) {
      el.setAttribute(attr, next);
    }
  }

  function renumberRow(row, fromNum, toNum) {
    if (fromNum === toNum) {
      return;
    }
    row.setAttribute("data-pathway", String(toNum));

    row.querySelectorAll("[id], [for], [name], [aria-describedby]").forEach(function (el) {
      ["id", "for", "name", "aria-describedby"].forEach(function (attr) {
        rewriteAttr(el, attr, fromNum, toNum);
      });
    });

    var title = row.querySelector(".card__title");
    if (title) {
      title.textContent = toNum === 1 ? "Pathway 1" : "Pathway " + toNum + " (optional)";
    }

    var remove = row.querySelector("[data-remove-pathway]");
    if (toNum === 1) {
      if (remove) {
        var wrap = remove.closest(".pathway-row__actions");
        if (wrap) {
          wrap.remove();
        } else {
          remove.remove();
        }
      }
    } else if (remove) {
      remove.textContent = "Remove pathway " + toNum;
    } else {
      var body = row.querySelector(".card__body");
      if (body) {
        var p = document.createElement("p");
        p.className = "pathway-row__actions";
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn btn--ghost btn--sm";
        btn.setAttribute("data-remove-pathway", "");
        btn.textContent = "Remove pathway " + toNum;
        p.appendChild(btn);
        body.appendChild(p);
      }
    }

    // Only pathway 1 keeps required name/organism.
    ["name", "organism"].forEach(function (base) {
      var input = row.querySelector("#" + base + "_" + toNum);
      if (!input) {
        return;
      }
      if (toNum === 1) {
        input.required = true;
      } else {
        input.removeAttribute("required");
      }
      var label = row.querySelector('label[for="' + base + "_" + toNum + '"]');
      if (label) {
        var req = label.querySelector(".field__req");
        if (toNum === 1 && !req) {
          var span = document.createElement("span");
          span.className = "field__req";
          span.textContent = "required";
          label.appendChild(span);
        } else if (toNum !== 1 && req) {
          req.remove();
        }
      }
    });
  }

  function renumberAll(form) {
    var rows = Array.prototype.slice.call(form.querySelectorAll(".pathway-row"));
    // Two-phase rename avoids colliding ids (e.g. 3→2 while 2 still exists).
    rows.forEach(function (row, idx) {
      var fromNum = parseInt(row.getAttribute("data-pathway"), 10);
      var tmp = 1000 + idx + 1;
      renumberRow(row, fromNum, tmp);
    });
    rows.forEach(function (row, idx) {
      var fromNum = parseInt(row.getAttribute("data-pathway"), 10);
      renumberRow(row, fromNum, idx + 1);
    });
  }

  function removePathway(form, row) {
    if (!row || row.getAttribute("data-pathway") === "1") {
      return;
    }
    clearInputs(row);
    row.setAttribute("hidden", "");
    var container = form.querySelector(".rows");
    if (container) {
      container.appendChild(row); // keep visible slots a contiguous prefix
    }
    renumberAll(form);
    syncAddButton(form);
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
        removePathway(form, remove.closest(".pathway-row"));
      }
    });

    syncAddButton(form);
  });
})();
