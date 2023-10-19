const images = data.near_image_files;

// Thêm hình ảnh vào nearImagesContainer trong tệp HTML
const nearImagesContainer = document.getElementById("nearImagesContainer");

images.forEach(image => {
    const imgElement = document.createElement("img");
    imgElement.src = `${image}`;
    imgElement.classList.add("responsive-near-image");
    imgElement.alt = "Near Image";
    
    const button = document.createElement("button");
    button.innerHTML = '<i class="fa fa-check" aria-hidden="true"></i>';
    button.classList.add("image-button-pick");

    const nearImgContainer = document.createElement("div");
    nearImgContainer.classList.add("nearImagesContainer");
    nearImgContainer.appendChild(imgElement);
    nearImgContainer.appendChild(button);
    nearImagesContainer.appendChild(nearImgContainer);
});
