%% Inverse Imaging Problem Solution
%% Step 1: Define the U-Net Architecture for Deblurring
function layers = createUNetDeblurring()
    layers = [
        imageInputLayer([256, 256, 1], 'Name', 'input', 'Normalization', 'none')

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')

        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')

        transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Name', 'deconv1')
        reluLayer('Name', 'relu3')

        transposedConv2dLayer(3, 1, 'Cropping', 'same', 'Name', 'deconv2')
        sigmoidLayer('Name', 'output')
    ];
end

%% Step 2: Load and Prepare the Input Image (X-ray)
function image = loadImage(imagePath)
    image = imread(imagePath); 

    % Convert RGB to grayscale if needed
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    image = im2double(image); % Convert to double precision in [0, 1]
    image = imresize(image, [256, 256]); % Resize to 256x256 for consistency
end

%% Step 3: Create a Blurred Image (Simulated for Deblurring)
function blurredImage = addBlur(image, blurKernelSize)
    % Create a Gaussian blur kernel
    h = fspecial('gaussian', blurKernelSize, blurKernelSize/6);
    blurredImage = imfilter(image, h, 'symmetric');
end

%% Main Script: Image Deblurring in Medical Images (X-ray)
% Get user input for the image path
imagePath = input('Enter the path to the blurry medical X-ray image: ', 's');

% Step 1: Load and Prepare the Original Image
originalImage = loadImage(imagePath);

% Step 2: Simulate a Blurry Image
blurredImage = addBlur(originalImage, 7);  % Simulating blur with a kernel size of 7x7
inputImage = blurredImage;  % Use the blurred image as input to the network

% Reshape for network input
inputImage = reshape(inputImage, [256, 256, 1]);

% Convert images to dlarray for deep learning
inputImage = dlarray(inputImage, 'SSCB'); % 'SSCB' refers to spatial (S), channel (C), and batch (B) dimensions
originalImage = dlarray(originalImage, 'SSCB');

% Define U-Net for Deblurring
layers = createUNetDeblurring();
net = dlnetwork(layerGraph(layers));  % Create a U-Net as a dlnetwork

% Random Noise for Input to Network (as in DIP)
inputNoise = randn([256, 256, 1], 'single');
inputNoise = dlarray(inputNoise, 'SSCB');

%% Step 4: Training Loop for Deblurring
maxEpochs = 1000;
learningRate = 1e-3;
trailingAvg = [];
trailingAvgSq = [];

% Prepare for GPU (if available)
if canUseGPU()
    inputNoise = gpuArray(inputNoise);
    inputImage = gpuArray(inputImage);
    originalImage = gpuArray(originalImage);
end

for epoch = 1:maxEpochs
    % Forward and Backward Pass
    [gradients, loss] = dlfeval(@modelGradients, net, inputNoise, originalImage, inputImage);
    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, epoch, learningRate);

    % Display progress every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

%% Display Results for Deblurring
% Get the output from the network (Enhanced Image)
deblurredImage = predict(net, inputNoise);
deblurredImage = extractdata(deblurredImage); % Extract data from dlarray
figure;
subplot(1, 2, 1); imshow(inputImage, []); title('Blurred Image (Input)');
subplot(1, 2, 2); imshow(deblurredImage, []); title('Deblurred Image (Output)');

%% Supporting Functions
function [gradients, loss] = modelGradients(net, inputNoise, originalImage, inputImage)
    % Forward pass
    output = forward(net, inputNoise);

    % Compute loss between the output (deblurred) and the original sharp image
    loss = mse(output, originalImage);

    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end
