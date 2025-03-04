ort.env.wasm.wasmPaths = './';


function dis(){
	 const imageInput = document.getElementById('imageInput');
    
    if (!imageInput.files.length) {
        outputDiv.innerText = "Please upload an image!";
        return;
    }

    const imageFile = imageInput.files[0];
    imageElement = new Image();
    const objectURL = URL.createObjectURL(imageFile);

    imageElement.src = objectURL;
    imageElement.onload = async () => {
       
            // Preprocess the image
            //const inputTensor = await preprocessImage(imageElement);
			
			div = document.getElementById('img')
	//div.innerHTML = '';
	
var parent = document.querySelector('#img');
while (parent.firstChild) {
        parent.removeChild(parent.firstChild);  
	}

    const canvas = document.createElement('canvas');
	
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(imageElement, 0, 0, 224, 224);
	
	//.value = canvas
	//var oldcanv = document.getElementById('canvas');
	div.appendChild(canvas);	
	
	}
}

async function preprocessImage(imageElement) {
    // Resize the image to 224x224
	//div.innerHTML = '';
	

    const canvas = document.createElement('canvas');
	
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(imageElement, 0, 0, 224, 224);
	
	
    // Extract image data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const { data } = imageData;

    // Preprocessing constants
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Prepare Float32Array for the model
    const inputTensor = new Float32Array(224 * 224 * 3);
    for (let i = 0; i < 224 * 224; i++) {
        inputTensor[i * 3] = (data[i * 4] / 255 - mean[0]) / std[0];     // R
        inputTensor[i * 3 + 1] = (data[i * 4 + 1] / 255 - mean[1]) / std[1]; // G
        inputTensor[i * 3 + 2] = (data[i * 4 + 2] / 255 - mean[2]) / std[2]; // B
    }

    // Reshape to [1, 3, 224, 224]
    const reshapedTensor = new Float32Array(1 * 3 * 224 * 224);
    for (let c = 0; c < 3; c++) {
        for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
                reshapedTensor[c * 224 * 224 + y * 224 + x] = inputTensor[y * 224 * 3 + x * 3 + c];
            }
        }
    }
    return reshapedTensor;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
}

async function runInference() {
    const imageInput = document.getElementById('imageInput');
    const outputDiv = document.getElementById('output');

    if (!imageInput.files.length) {
        outputDiv.innerText = "Please upload an image!";
        return;
    }


        try {
            // Preprocess the image
            const inputTensor = await preprocessImage(imageElement);

            // Load ONNX model
            const session = await ort.InferenceSession.create('./model.onnx');

            // Prepare the input
            const inputName = session.inputNames[0]; // 'input'
            const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 224, 224]);

            // Run inference
            const feeds = { [inputName]: tensor };
            const results = await session.run(feeds);

            // Get output
            const outputName = session.outputNames[0]; // 'output'
            const logits = results[outputName].data;
            const probabilities = softmax(Array.from(logits));

            // Get class index and confidence
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = (probabilities[maxIndex] * 100).toFixed(2);

            // Fetch labels
            const labels = await fetch('./labels.txt').then(res => res.text());
            const labelArray = labels.split('\n');
			
			lab = labelArray[maxIndex]
			cof = confidence
            // Display result
			sent()
            
        } catch (error) {
            console.error("Error running inference:", error);
            outputDiv.innerText = "Error running inference. Check console for details.";
        }

}

function sent(){
	    const outputDiv = document.getElementById('output');
		labe = lab.split("__")
		laben = labe[1]
		labin = laben.trim()
		laba = labe[0]
		if (cof<50){
	    outputDiv.innerText = 'Unclassified Image, Try with Clearer Image!! ' 
		} else if(labin == 'Healthy') {
			dise = 'NONE' +'\n'
			stater = 'Healthy'
		} else {
			stater = 'Diseased'
			dise = laben
		}
			
	outputDiv.innerText = 'Crop Type: ' + laba + '\n' + ' State: '+ stater + '\n'+ ' Disease Type: '+ dise + 'Confidence: ' + cof +'%';

}