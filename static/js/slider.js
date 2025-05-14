window.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.comparison-slider').forEach(function(slider) {
    var container = slider.closest('.comparison');
    var handle    = container.querySelector('.handle');
    var divisor   = container.querySelector('.divisor');

    function move() {
      var pct = slider.value + '%';
      handle.style.left = pct;
      var inv = 100 - slider.value;
      divisor.style.clipPath = 'inset(0 ' + inv + '% 0 0)';
      divisor.style.webkitClipPath = 'inset(0 ' + inv + '% 0 0)';
      container.style.setProperty('--pct', pct);
    }
    slider.addEventListener('input', move);
    move();
  });
});