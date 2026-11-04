// Part nine: the scale and the first load.
  function applyScale() {
    if (!face) return;

    // The card, not the window: in a host the faceplate is inset, and the
    // controls have to fit the card they sit on.
    var card = app.querySelector(".face");
    var width = (card ? card.clientWidth : app.clientWidth) || face.size[0];
    var height = (card ? card.clientHeight : app.clientHeight) || face.size[1];
    scale = clamp(Math.min(width / face.size[0], height / face.size[1]), 0.8, 1.9);
    document.documentElement.style.setProperty("--scale", scale);
  }

  window.addEventListener("resize", function () {
    requestAnimationFrame(applyScale);
  });

  // A host resizes the WebView2 child, not the browser window: the window event
  // is not guaranteed there, but the app's own box always changes.
  if (window.ResizeObserver) {
    new ResizeObserver(function () {
      requestAnimationFrame(applyScale);
    }).observe(app);
  }

  fetch("/face", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("face " + response.status);
      return response.json();
    })
    .then(function (page) {
      face = page;
      wantsAutotune = page.sections.some(function (section) {
        return section.hero === "autotune";
      });
      buildFace();
      pollAllowed();
    })
    .catch(function (error) {
      app.textContent = "faceplate unavailable: " + error;
      console.error("i_am_web_gui: " + error);
    });
})();
