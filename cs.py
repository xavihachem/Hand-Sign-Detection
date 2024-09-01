import cv2

image_path = "C:\\Users\\ash56\\Githu\\Hand-Sign-Detection\\image.png"
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error to load image")
else:
    gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_scale, 50 , 150)

    cv2.imshow('Original image',original_image)
    cv2.imshow('grayscale image',gray_scale)
    cv2.imshow('edges',edges)

    cv2.waitKey(0)
    cv2.destroyWindow()