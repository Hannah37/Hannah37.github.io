(function () {
	var header = document.getElementById("mainHeader");

	function changeHeader() {
		var scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
		header.classList.toggle("header-background", scrollTop >= 50 || document.body.classList.contains("nav-open"));
	}

	var didScroll = false;

	$(window).scroll(function () {
		didScroll = true;
	});

	setInterval(function() {
		if (didScroll) {
			didScroll = false;
			changeHeader();
		}
	}, 100);

	changeHeader();

	document.getElementById("open-nav").addEventListener("click", function (event) {
		event.preventDefault();
		document.body.classList.toggle("nav-open");
		changeHeader();
	});

	$("a[href*=\\#]").on("click", function (event) {
		if(this.pathname === window.location.pathname) {
			event.preventDefault();
			
			var targetId = this.hash.replace(/#/, '')
			var elem = document.getElementById(targetId)
			var targetOffset = $(elem).offset().top
			var sourceOffset = $(document).scrollTop()

			// if going up, take account navbar
			if (targetOffset < sourceOffset) {
				var navbarHeight = $('header').outerHeight()
				targetOffset = targetOffset - navbarHeight
			}

			$("html, body").animate({
				scrollTop: targetOffset
			}, 500);
		}
	});
})();