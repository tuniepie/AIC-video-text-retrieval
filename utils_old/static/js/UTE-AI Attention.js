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
//---------------------------------------------------------------------------------
class Button {
  constructor(idButton, ...buttonFunctions) {
    const xButton = document.getElementById(idButton);
    xButton.addEventListener("click", () => {
      buttonFunctions.forEach(func => func());
    });
  }
}
// testFunction--------------------------------------------------------------------
function showAlert(message) {
  alert(message);
}

function openSearch() {
  window.open("search.html", "_blank");
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
  }
}
// functions--------------------------------------------------------------------
function saveData(idInputField) {
  var inputField = document.getElementById(idInputField).value;
  sessionStorage.inputField = inputField;
}

function clearSearch(idInputField) {
  const inputField = document.getElementById(idInputField);
  inputField.value = "";  
}

// Hàm để hiển thị imageKeyFrame
function displayImageKeyFrame(keyFrame) {
  const keyFrameContainer = document.getElementById("BLIP");
  const keyFrameElement = document.createElement("p");
  keyFrameElement.textContent = "Key Frame: " + keyFrame;
  keyFrameElement.classList.add("key-frame-element"); //id này dùng để reset
  keyFrameContainer.appendChild(keyFrameElement);
}

// Hàm để xóa imageKeyFrame khỏi màn hình
function removeImageKeyFrame() {
  const keyFrameContainer = document.getElementById("BLIP");
  const keyFrameElement = keyFrameContainer.querySelector("p");
  if (keyFrameElement) {
    keyFrameContainer.removeChild(keyFrameElement);
  }
}

// Hàm để cập nhật trạng thái của button1
function updateButtonState(button, isClicked) {
  if (isClicked) {
    button.classList.add("button1-clicked");
  } else {
    button.classList.remove("button1-clicked");
  }
}

// Hàm tạo CSV từ các ảnh đc chọn
function createCSVContent() {
  const selectedFrames = [];
  for (let i = 0; i < imageKeyFrameSelected.length; i++) {
    if (imageKeyFrameSelected[i]) {
      selectedFrames.push(imagesKeyFrame[i]);
    }
  }
  return selectedFrames.join('\n');
}

// Hàm tải file CSV
function downloadCSV() {
  const csvContent = createCSVContent();
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = 'selected_frames.csv';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
}

//Hàm nhận key frames liên quan
// function nearKeyFrames(imageKeyFrame) {
//   const serverUrl = 'http://127.0.0.1:8282/nearKeyFrames';
//   fetch(serverUrl, {
//     method: 'POST',
//     body: JSON.stringify({ imageKeyFrame }), 
//     headers: {
//       'Content-Type': 'application/json'
//     }
//   })
//   .then(response => response.json())
//   .then(data => {
//     //
//   })
//   .catch(error => console.error(error));
// }

//Hàm search bằng BLIP
var imagesKeyFrame = [];
var buttonStates = [];
var imageKeyFrameSelected = [];

function search(idSearch) {
  var idSearch = document.getElementById(idSearch).value;
  const imageContainer = document.getElementById("image-container");
  while (imageContainer.firstChild) {
    imageContainer.removeChild(imageContainer.firstChild);
  }
  var serverUrl = 'http://127.0.0.1:8281/search';
  fetch(serverUrl, {
    method: 'POST',
    body: JSON.stringify({input: idSearch}),
    headers: {
      'Content-Type': 'application/json'
    }    
  })
  .then(response => response.json())
  .then(data => {
    data.image_files.forEach((image,index) => {
        const imgElement = document.createElement("img");
        imgElement.src = `./static/${image}`;
        imgElement.classList.add("responsive-image");

        //Button tạo file csv để ghi key_frames
        const button1 = document.createElement("button");
        button1.innerHTML = '<i class="fa fa-check" aria-hidden="true"></i>';
        button1.classList.add("image-button-pick");
        imagesKeyFrame[index] = data.key_frames[index];
        buttonStates[index] = false;
        imageKeyFrameSelected[index] = false;
        // process button1 event
        button1.addEventListener("click", function() {
          var imageKeyFrame = imagesKeyFrame[index];
          if (imageKeyFrameSelected[index]) {
            imageKeyFrameSelected[index] = false;
            removeImageKeyFrame();
          } else {
            imageKeyFrameSelected[index] = true;
            displayImageKeyFrame(imageKeyFrame);
          }
          buttonStates[index] = imageKeyFrameSelected[index];
          updateButtonState(button1, buttonStates[index]);
        });


        //Button tìm các frame xung quanh
        const button2 = document.createElement("button");
        button2.innerHTML = '<i class="fa fa-plus" aria-hidden="true"></i>';
        button2.classList.add("image-button-add");
        // process button2 event
        button2.addEventListener("click", function() {
          const imageKeyFrame = imagesKeyFrame[index];
          // gửi imageKeyFrame để nhận lại relatedKeyFrame

          //mở trang mới và hiển thị relatedKeyFrames
          const newPage = window.open("about:blank", "_blank");
          newPage.onload = function() {
            const pElement = document.createElement("p");
            pElement.textContent = imageKeyFrame;
            newPage.document.body.appendChild(pElement);
          };

        })

        const imgContainer = document.createElement("div");
        imgContainer.classList.add("image-container");    
        imgContainer.appendChild(imgElement);
        imgContainer.appendChild(button1);
        imgContainer.appendChild(button2);    
        imageContainer.appendChild(imgContainer);
      });
  })
  .catch(error => console.error(error));
}

function clearElements() {
  imageKeyFrameSelected = [];

  const displayedKeyFrames = document.querySelectorAll(".key-frame-element");
  displayedKeyFrames.forEach((keyFrameElement) => {
    keyFrameElement.remove();
  });

  const buttons = document.querySelectorAll(".button1-clicked");
  buttons.forEach((button) => {
    button.classList.remove("button1-clicked");
  });

}
//---------------------------------------------------------------------------------
function init() {
  var defaultOpen = document.getElementById("defaultOpen");
  defaultOpen.onclick = function() {
    openPage(defaultOpen.dataset.page, defaultOpen, defaultOpen.dataset.color);
  };
  
  document.getElementById("defaultOpen").click();

// BLIP===========================================================================
  const searchBLIPButton = new Button("searchBLIPButton", () => {
    saveData("inputBLIP");
    search("inputBLIP");
  });

  const clearBLIPButton = new Button("clearBLIPButton", () => {
    clearSearch("inputBLIP");
    clearElements();
  });

  const voiceBLIPButton = new Button("voiceBLIPButton", () => {
    alert("Voice...");
  })

  const downloadBLIPButton = new Button("downloadBLIPButton", () => {
    downloadCSV();
  })
// CLIP===========================================================================
  const searchCLIPButton = new Button("searchCLIPButton", () => {
    saveData("inputCLIP");
    openSearch();
  });

  const voiceButton = new Button("voiceCLIPButton", () => {
    alert("Voice...");
  })

  const clearCLIPButton = new Button("clearCLIPButton", () => {
    clearSearch("inputCLIPField");
  });

  const downloadCLIPButton = new Button("downloadCLIPButton", () => {
    alert('DOne');
  })
// Intern Video===========================================================================
  const searchInVButton = new Button("searchInVButton", () => {
    saveData("inputInVField");
    openSearch();
  });

  const voiceInVButton = new Button("voiceInVButton", () => {
    alert("Voice...");
  })

  const clearInVButton = new Button("clearInVButton", () => {
    clearSearch("inputInVField");
  });

  const downloadInVButton = new Button("downloadInVButton", () => {
    alert('DOne');
  })
}
//===========================================================================
document.addEventListener("DOMContentLoaded", () => {
  window.onload = init;
});
