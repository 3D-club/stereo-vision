import numpy as np
import cv2
import glob

nffile = np.load('D:/3d workshop/Telegram Desktop/Camera.npz')

mtx = nffile['mtx']

dist =  nffile['dist']




cap = cv2.VideoCapture(0)
ret,img=cap.read()
h,  w = img.shape[:2]
cv2.imshow("Original",img)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))


mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst1 = dst[y:y+h, x:x+w]
#cv2.imwrite('right1.png', dst1)
cv2.imwrite('left1.png', dst1)

cv2.imshow('cal',dst1)
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()
