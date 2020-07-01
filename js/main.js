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

			$("html, body").animate({
				scrollTop: $(this.hash).offset().top
			}, 500);
		}
	});


$(document).ready(() => {
    const contactForm = $(".contact-form");
    const contactFormMethod = contactForm.attr("method");
    const contactFormEndpoint = contactForm.attr("action");
  
  
    function displaySubmitting(submitBtn, defaultText, doSubmit) {
      // const defaultText = submitBtn.text();
      if (doSubmit) {
        submitBtn.addClass("disabled");
        submitBtn.html(
          "<i class='fa fa-spin fa-spinner'></i> Sending..."
        );
      } else {
        submitBtn.removeClass("disabled");
        submitBtn.html(defaultText);
      }
    }
  
    contactForm.submit(function(e) {
      e.preventDefault();
      const contactFormSubmitBtn = contactForm.find("[type='submit']");
      const contactFormSubmitBtnTxt = contactFormSubmitBtn.text();
      const contactFormData = contactForm.serialize();
      // const thisForm = $(this);
      displaySubmitting(contactFormSubmitBtn, '', true);
      $.ajax({
        url: contactFormEndpoint,
        method: contactFormMethod,
        data: contactFormData,
        success: function(data) {
          // thisForm[0].reset();
          contactForm[0].reset();
          $.alert({
            title: "Success!",
            content: data.message,
            theme: "supervan"
          });
          setTimeout(() => {
            displaySubmitting(contactFormSubmitBtn, contactFormSubmitBtnTxt, false);
          }, 500);
        },
        error: function(error) {
          const jsonData = error.responseJSON;
          let msg = "";
          $.each(jsonData, (key, value) => {
            msg += key + ": " + value[0].message + "<br>";
          });
          $.alert({
            title: "Oops!",
            content: msg,
            theme: "modern"
          });
          console.log("Error when saving contact form => ", error);
          setTimeout(() => {
            displaySubmitting(contactFormSubmitBtn, contactFormSubmitBtnTxt, false);
          }, 500);
        }
      });
    });
  });
})();
