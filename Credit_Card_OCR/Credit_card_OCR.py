# pip install --upgrade imutils
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
# argparse란? https://blog.naver.com/qbxlvnf11/221801181428
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-r", "--reference", required=True, help="path to reference OCR-A image")
args = vars(ap.parse_args())

# define a dictionary that maps the first digit of a credit cart
# number to the credit card type
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# load the reference OCR-A image from disk, convert it to grayscale.
# and threshold it, such that the digits appear as *white* on a *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("ref", ref)