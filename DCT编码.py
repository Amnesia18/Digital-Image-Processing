import cv2
import numpy as np
import heapq
import os
from collections import defaultdict

# Function to perform Run-Length Encoding (RLE)
def run_length_encoding(image_array):
    flat_array = image_array.flatten()
    current = flat_array[0]
    count = 1
    encoded = []
    for pixel in flat_array[1:]:
        if pixel == current:
            count += 1
        else:
            encoded.append((current, count))
            current = pixel
            count = 1
    # Adding the last run
    encoded.append((current, count))
    return encoded

# Function to calculate frequency of each pixel value in the image
def calculate_frequency(image_array):
    freq = defaultdict(int)
    for pixel in image_array.flatten():
        freq[pixel] += 1
    return freq

# Function to build Huffman Tree
def build_huffman_tree(freq):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Function to encode the image using Huffman coding
def huffman_encode(image_array, huffman_tree):
    huffman_code = {symbol: code for symbol, code in huffman_tree}
    encoded_output = []
    for pixel in image_array.flatten():
        encoded_output.append(huffman_code[pixel])
    return ''.join(encoded_output), huffman_code

# Function to perform DCT compression using JPEG
def jpeg_compression(image, quality, path):
    # Save the image as JPEG with the given quality
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
    # Get the size of the compressed image
    compressed_size = os.path.getsize(path)
    return compressed_size

# Function to calculate compression ratio
def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# Load and process the image
image_path = 'digital signal processing/rice.tif'  # Update this path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Original size of the image
original_size = original_image.nbytes  # Size in bytes

# RLE Compression
rle_encoded = run_length_encoding(original_image)
rle_size = len(rle_encoded) * 2 * 4  # Each RLE entry has two parts (value, count), each 4 bytes (int32)
rle_compression_ratio = compression_ratio(original_size, rle_size)

# Huffman Compression
pixel_freq = calculate_frequency(original_image)
huffman_tree = build_huffman_tree(pixel_freq)
huffman_encoded, huffman_code = huffman_encode(original_image, huffman_tree)
huffman_size = len(huffman_encoded) // 8  # Convert from bits to bytes
huffman_compression_ratio = compression_ratio(original_size, huffman_size)

# DCT Compression (JPEG)
jpeg_quality = 50  # Medium quality
jpeg_path = 'digital signal processing/rice.tif'  # Update this path
jpeg_size = jpeg_compression(original_image, jpeg_quality, jpeg_path)
dct_compression_ratio = compression_ratio(original_size, jpeg_size)

# Output the results
print(f"Original size: {original_size} bytes")
print(f"RLE compressed size: {rle_size} bytes, Compression ratio: {rle_compression_ratio}")
print(f"Huffman compressed size: {huffman_size} bytes, Compression ratio: {huffman_compression_ratio}")
print(f"DCT (JPEG) compressed size: {jpeg_size} bytes, Compression ratio: {dct_compression_ratio}")
