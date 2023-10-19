// openTabs----------------------------------------------------------------------
function openPage(pageName, elmnt, color) {
  var tabcontent = document.getElementsByClassName("tabcontent");
  for (var i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  
  var tablinks = document.getElementsByClassName("tablink");
  for (var i = 0; i < tablinks.length; i++) {
    tablinks[i].style.backgroundColor = "";
  }
  
  document.getElementById(pageName).style.display = "block";
  elmnt.style.backgroundColor = color;
}

// testFunction--------------------------------------------------------------------
function showAlert(message) {
  alert(message);
}

function saveData(idInputField) {
	var inputField = document.getElementById(idInputField).value;
	sessionStorage.inputField = inputField;
}

function openSearch() {
  window.open("search.html", "_blank");
}

function clearSearch(idInputField) {
  const inputField = document.getElementById(idInputField);
  inputField.value = "";  
}

function displaySelectedImage(file) {
  var selectedImage = document.getElementById('selectedImage');
  selectedImage.style.display = 'block';
  selectedImage.src = URL.createObjectURL(file);
}

function clearSelectedImage() {
  var selectedImage = document.getElementById('selectedImage');
  selectedImage.style.display = 'none';
  selectedImage.src = '';
}

function handleImageUpload() {
  var imageInput = document.getElementById('imageInput');
  var selectedFile = imageInput.files[0];

  if (selectedFile) {
      displaySelectedImage(selectedFile);

      var formData = new FormData();
      formData.append('image', selectedFile);

      // Gửi dữ liệu formData đến máy chủ bằng cách sử dụng API fetch() hoặc AJAX
      // Ví dụ sử dụng fetch():
      /*
      fetch('url_to_server', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          console.log('Upload successful:', data);
      })
      .catch(error => {
          console.error('Upload failed:', error);
      });
      */
  }
}

//---------------------------------------------------------------------------------
class Button {
  constructor(idButton, ...buttonFunctions) {
    const xButton = document.getElementById(idButton);
    xButton.addEventListener("click", () => {
      buttonFunctions.forEach(func => func());
    });
  }
}

//---------------------------------------------------------------------------------
function init() {
  var defaultOpen = document.getElementById("defaultOpen");
  defaultOpen.onclick = function() {
    openPage(defaultOpen.dataset.page, defaultOpen, defaultOpen.dataset.color);
  };
  
  document.getElementById("defaultOpen").click();

// Tab_1===========================================================================
  const searchTextButton = new Button("searchTextButton", () => {
    saveData("inputTextField");
    openSearch();
  });

  const clearTextButton = new Button("clearTextButton", () => {
    clearSearch("inputTextField");
  });

  const submitTextButton = new Button("submitTextButton", () => {
    alert("Done!");
  })
// Tab_2===========================================================================
  const searchASRButton = new Button("searchASRButton", () => {
    saveData("inputASRField");
    openSearch();
  });

  const clearASRButton = new Button("clearASRButton", () => {
    clearSearch("inputASRField");
  });

  const voiceButton = new Button("voiceButton", () => {
    alert("Voice...");
  })

  const submitASRButton = new Button("submitASRButton", () => {
    alert("Done!");
  })
// Tab_3===========================================================================
  const searchOCRButton = new Button("searchOCRButton", () => {
    saveData("inputOCRField");
    openSearch();
  });

  const clearOCRButton = new Button("clearOCRButton", () => {
    clearSearch("inputOCRField");
  });

  const submitOCRButton = new Button("submitOCRButton", () => {
    alert("Done!");
  })
// Tab_4===========================================================================
const imageButton = new Button("imageButton", () => {
  document.getElementById('imageInput').click();
})

document.getElementById('imageInput').addEventListener('change', function() {
  handleImageUpload();
});

const searchImageButton = new Button("searchImageButton", () => {
  alert("Done!");
});

const clearImageButton = new Button("clearImageButton", () => {
  clearSelectedImage();
});

const submitImageButton = new Button("submitImageButton", () => {
  alert("Done!");
})
}

document.addEventListener("DOMContentLoaded", () => {
  window.onload = init;
});


