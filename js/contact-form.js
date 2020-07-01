// Helper function to get form data in the supported format
function getFormDataString(formEl) {
  var formData = new FormData(formEl),
      data = [];

  for (var keyValue of formData) {
    data.push(encodeURIComponent(keyValue[0]) + "=" + encodeURIComponent(keyValue[1]));
  }

  return data.join("&");
}

// Fetch the form element
var formEl = document.getElementById("contact-form");

// Override the submit event
formEl.addEventListener("submit", function (e) {
  // if(document.getElementById('name').value.length === 0){
  //   alert('Name 필드의 값이 누락 되었습니다');
  //   e.preventDefault();
  // }

  // if (grecaptcha) {
  //   var recaptchaResponse = grecaptcha.getResponse();
  //   if (!recaptchaResponse) { // reCAPTCHA not clicked yet
  //     return false;
  //   }
  // }

  console.log("\nforme1.method: ", formEl.method, formEl.action);
  var request = new XMLHttpRequest();
  if(!request) {
    alert('XMLHTTP 인스턴스를 만들 수가 없어요 ㅠㅠ');
    return false;
  }
  else {
    console.log("\nrequest status is : ", request.status);
  }


  // request.addEventListener("load", function () {
    
  //   if (request.status === 302) { // CloudCannon redirects on success
  //     // It worked
  //     console.log("\nSent Message\n");
  //   }
  // });

  request.open(formEl.method, formEl.action);
  request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  request.send(getFormDataString(formEl));
});