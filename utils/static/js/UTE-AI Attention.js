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

class Key {
  constructor(idInput, pressedKey, ...keyFunctions) {
    const xKey = document.getElementById(idInput);
    xKey.setAttribute('keypress', pressedKey);
    xKey.addEventListener("keydown", function(event) {
      if (event.key === pressedKey) {
        keyFunctions.forEach(func => func());
      }
    });
  }
}
// speech to text function------------------------------------------------------
function checkMicrophonePermission() {
  const hasPermission = localStorage.getItem('microphonePermission') === 'granted';
  if (hasPermission) {
    setupSpeechRecognition();
  } else {
    // Disable the microphone button initially.
    document.getElementById("voiceBLIPButton").disabled = true;
    requestMicrophoneAccess();
  }
}


function requestMicrophoneAccess() {
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then(function (stream) {
      setupSpeechRecognition();
      // Save microphone permission in Local Storage to avoid requesting again.
      localStorage.setItem('microphonePermission', 'granted');
      // Enable the microphone button.
      document.getElementById("voiceBLIPButton").disabled = false;
    })
    .catch(function (error) {
      console.error('Error when requesting microphone access:', error);
    });
}


function setupSpeechRecognition() {
  const voiceBLIPButton = document.getElementById("voiceBLIPButton");
  const inputBLIP = document.getElementById("inputBLIP");
  const recognition = new webkitSpeechRecognition();

  // Disable nút voiceBLIPButton trong lúc thu âm
  voiceBLIPButton.disabled = true;
  voiceBLIPButton.innerHTML = '<i class="fa fa-headphones" aria-hidden="true"></i>';

  recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    inputBLIP.value = transcript;
    inputBLIP.focus();

    // Khi hoàn thành việc thu âm, bật lại nút voiceBLIPButton
    voiceBLIPButton.disabled = false;
    voiceBLIPButton.innerHTML = '<i class="fa fa-microphone" aria-hidden="true"></i>';
  };

  recognition.onend = function() {
    // Khi kết thúc thu âm (ngừng microphone), bật lại nút voiceBLIPButton
    voiceBLIPButton.disabled = false;
    voiceBLIPButton.innerHTML = '<i class="fa fa-microphone" aria-hidden="true"></i>';
  };

  recognition.start();
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


// Hàm tải file CSV
function submit(imageKeyFrameSelected) {
  // var inputString = "L05_V010,025464";
  var parts = imageKeyFrameSelected.split(",");

  // Construct the data to send as JSON

  var VIDEO_ID = parts[0];
  var FRAME_ID = parts[1];
  var SESSION_ID = "node0fbcpvh59k7e21esqfbfw105fq2849"; // Replace with your actual session ID
// Build URL with parameters
  const url = `https://eventretrieval.one/api/v1/submit?item=${VIDEO_ID}&frame=${FRAME_ID}&session=${SESSION_ID}`;
  const params = {
    item: VIDEO_ID,
    frame: FRAME_ID,
    session: SESSION_ID
};

  // Perform a GET request to send the request to the API
  fetch(url, { method: 'GET' })
      .then(response => {
          if (response.status === 200) {
              return response.json();
          } else {
              console.log(`Failed to fetch data. Status code: ${response.status}`);
              throw new Error(`Failed to fetch data. Status code: ${response.status}`);
          }
      })
      .then(data => {
          const description = data.description;
          const submission_status = data.status;
          console.log(`Description: ${description}`);
          console.log(`Submission Status: ${submission_status}`);
      })
      .catch(error => {
          console.error(error);
      });
    }

function openNearImages(data) {
  // CSS cho các ảnh
    const imageStyle = `
    width: 30%;
    margin: 0.5%;
    height: auto;
  `;

  const containerStyle = `
    position: relative;
    display: inline-flex;
    padding: 3px;
    width: 30%;
    max-width: 30%;
    height: auto;
  `;

  const buttonStyle = `
    position: absolute;
    top: 2.5%; 
    left: 1.5%; 
    height: 15%; 
    width: 15%; 
    background: #ddd;
    border: none;
    cursor: pointer;
    z-index: 1;
    font-size: 50%;
  `;

  const nearImagesHtml = `
    <html>
    <head>
      <title>UTE-AI Attention</title>
      <style>
        .nearImagesContainer {
          display: flex;
          flex-wrap: wrap;
        }
        .nearImagesContainer .nearImagesItem {
          ${containerStyle}
        }
        .nearImagesContainer img {
          ${imageStyle}
        }
        .imageButton {
          ${buttonStyle}
        }
      </style>
    </head>
    <body>
      <div id="nearImagesContainer" class="nearImagesContainer">
      ${data.map((image, index) => `
        <div class="nearImagesItem">
          <img src="${image}" alt="Near Image" style="${imageStyle}" />
          <button class="imageButton" onclick="alert('Button clicked for image ${index + 1}')">
            <i class="fa fa-plus"></i>
          </button>
        </div>
      `).join('')}
      </div>
    </body>
    </html>
  `;

  // Tạo một Blob từ HTML
  const blob = new Blob([nearImagesHtml], { type: 'text/html' });
  const url = window.URL.createObjectURL(blob);
  const newWindow = window.open(url, '_blank');
  window.URL.revokeObjectURL(url);
}




//Hàm nhận key frames liên quan
function nearKeyFrames(clickedImageId) {
  const serverUrl = 'http://127.0.0.1:8210/nearKeyFrames';
  fetch(serverUrl, {
    method: 'POST',
    body: JSON.stringify({ clickedImageId: clickedImageId }), 
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);
    const imageWidth = "auto"; 
    const imageHeight = "auto";
    openNearImages(data.near_image_files);
  })
  .catch(error => console.error(error));
}



//Hàm search bằng BLIP
var imagesKeyFrame = [];
var buttonStates = [];
var imageKeyFrameSelected = [];
var imageId = [];
var gimageKeyFrame = '';

function search(idSearch) {
  var idSearch = document.getElementById(idSearch).value;
  const imageContainer = document.getElementById("image-container");
  while (imageContainer.firstChild) {
    imageContainer.removeChild(imageContainer.firstChild);
  }
  var serverUrl = 'http://127.0.0.1:8210/search';
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
        removeImageKeyFrame();
        const imgElement = document.createElement("img");
        imgElement.src = `./static/${image}?q=0.5`;
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
          gimageKeyFrame = imageKeyFrame
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
        imageId[index] = data.images_id[index];
        // process button2 event
        button2.addEventListener("click", function() {
          const clickedImageId = imageId[index];
          nearKeyFrames(clickedImageId);
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

  const searchBLIPKey = new Key("inputBLIP", "Enter", () => {
    saveData("inputBLIP");
    search("inputBLIP");
  });  

  const clearBLIPButton = new Button("clearBLIPButton", () => {
    clearSearch("inputBLIP");
    clearElements();
  });

  const voiceBLIPButton = new Button("voiceBLIPButton", () => {
    setupSpeechRecognition();
  });

  const downloadBLIPButton = new Button("downloadBLIPButton", () => {
    downloadCSV();
  });

  const submitBLIPButton = new Button("submitBLIPButton", () => {
    submit(gimageKeyFrame)
    // alert("Sent!");
  });
// CLIP===========================================================================
  const searchCLIPButton = new Button("searchCLIPButton", () => {
    saveData("inputCLIP");
    openSearch();
  });

  const voiceButton = new Button("voiceCLIPButton", () => {
    alert("Voice...");
  });

  const clearCLIPButton = new Button("clearCLIPButton", () => {
    clearSearch("inputCLIPField");
  });

  const downloadCLIPButton = new Button("downloadCLIPButton", () => {
    alert('DOne');
  });

  const submitCLIPButton = new Button("submitCLIPButton", () => {
    alert("Sent!");
  });
// Intern Video===========================================================================
  const searchInVButton = new Button("searchInVButton", () => {
    saveData("inputInVField");
    openSearch();
  });

  const voiceInVButton = new Button("voiceInVButton", () => {
    alert("Voice...");
  });

  const clearInVButton = new Button("clearInVButton", () => {
    clearSearch("inputInVField");
  });

  const downloadInVButton = new Button("downloadInVButton", () => {
    alert('DOne');
  });

  const submitInVButton = new Button("submitInVButton", () => {
    alert("Sent!");
  });

  window.addEventListener('load', checkMicrophonePermission);
}
//===========================================================================
document.addEventListener("DOMContentLoaded", () => {
  window.onload = init;
});
