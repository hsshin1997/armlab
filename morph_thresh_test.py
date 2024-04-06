import cv2

thresh = cv2.imread("morph_thresh.png")
thresh_prev = cv2.imread("morph_thresh.png")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("morph_thresh_processed.png", thresh)

thresh_diff = thresh - thresh_prev
cv2.imwrite("morph_thresh_diff.png", thresh_diff)

# _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

a = 1