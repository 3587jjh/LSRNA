window.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.comparison-slider').forEach(function(slider) {
    var container = slider.closest('.comparison');
    var handle    = container.querySelector('.handle');
    var divisor   = container.querySelector('.divisor');

    function move() {
      var pct = slider.value + '%';
      handle.style.left   = pct;
      divisor.style.width = pct;
    }

    slider.addEventListener('input', move);
    move();
  });
});