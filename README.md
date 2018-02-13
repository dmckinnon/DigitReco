# DigitReco

Aims:
1. Use tensorflow to recognise handwritten digits
2. Use OpenCV to get a bounding box around handwritten digits in an image
  2a) Use B-Human's algorithm to find the optimal threshold point, and binarise the image
  2b) Dilate the black
  2c) Find black regions, find the bounding box (watershed/floodfill)
  2d) this gets a box around each digit
3. Classify these digits, and replace within the bounding box with a neatly typed digit, and also store and print the string. keeping formatting like new lines, etc

4. Long term goal - extend this to the alphabet, and be able to convert handwritten pages to text
