// Part nine: the panel switch, a repaint and the first load.
  function showPanel(name) {
    if (!panelPages) return;
    activePanel = name;
    panelTabs.forEach(function (tab) {
      tab.classList.toggle("on", tab.getAttribute("data-panel") === name);
    });
    panelBody.textContent = "";
    panelBody.appendChild(panelPages[name]);

    if (name === "sources") {
      var sources = panelBody.querySelector(".mod-sources");
      if (sources) sources.scrollLeft = sourceScroll;
    }
  }

  function render(data) {
    current = data;
    rangeLabels = [];
    livePainters = [];
    liveCards = [];
    sourceNames = Object.create(null);
    sourceColors = Object.create(null);

    var counts = Object.create(null);
    var seen = Object.create(null);

    (data.sources || []).forEach(function (source) {
      counts[source.kind] = (counts[source.kind] || 0) + 1;
    });

    (data.sources || []).forEach(function (source) {
      seen[source.kind] = (seen[source.kind] || 0) + 1;
      sourceNames[source.id] =
        source.name + (counts[source.kind] > 1 ? " " + seen[source.kind] : "");
      sourceColors[source.id] = sourceAccent(source.kind);
    });

    (data.macro_names || []).forEach(function (name, index) {
      sourceNames[4 + index] = name || ("M" + (index + 1));
      sourceColors[4 + index] = "#9a9aff";
    });
    pageRate = data.rate;
    rack.textContent = "";

    rack.appendChild(makeCrossover(data));
    data.bands.forEach(function (band, index) {
      rack.appendChild(makeBand(data, index));
    });

    rack.querySelectorAll(".band .slots").forEach(function (element, index) {
      element.scrollLeft = slotScroll[index] || 0;
    });

    panelPages = {
      sources: makeSourcesPage(data),
      matrix: makeMatrix(data),
      daw: makeAutomation(data)
    };
    showPanel(activePanel);

    updateRanges(data);

    var mods = (data.assignments || []).filter(function (assignment) {
      return assignment;
    }).length;

    rate.textContent = (data.rate / 1000).toFixed(1) + " kHz";
    status.textContent = data.modules.length + " modules \u00B7 " + data.bands.length +
      " lanes \u00B7 " + mods + " mods";
    refreshEditSlots(data);
  }

  function load() {
    var top = rack.scrollTop;

    return fetch("/rack", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("rack " + response.status);
        return response.json();
      })
      .then(function (data) {
        render(data);
        rack.scrollTop = top;
      })
      .catch(function (error) {
        status.textContent = "rack unavailable: " + error.message;
      });
  }
