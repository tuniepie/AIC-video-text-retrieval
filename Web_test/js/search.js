function loadData() {
	var inputField = document.getElementById("inputField");

	inputField.textContent = sessionStorage.inputField;
}

function init() {
    loadData();
}

window.onload = init;