let model;
let labels = [];

window.onload = async () => {
  model = await tf.loadLayersModel("model.json");
  const res = await fetch("labels.json");
  labels = await res.json();
};

document.getElementById("imageInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("preview").src = event.target.result;
    };
    reader.readAsDataURL(file);
  }
});

async function predict() {
  const img = document.getElementById("preview");
  if (!img.src) return;

  const tensor = tf.browser
    .fromPixels(img)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(tf.scalar(255))
    .expandDims();

  const predictions = await model.predict(tensor).data();
  const topIdx = predictions.indexOf(Math.max(...predictions));
  const breed = labels[topIdx];

  document.getElementById("predictionResult").innerText = `Predicted: ${breed}`;
}
