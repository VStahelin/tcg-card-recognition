from hashing import convert_hash
from hashing import hamming
from hashing import dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2

IMAGES = "photos/archive/all/"
TREE = "hash/vptree.pickle"
HASHES = "hash/hashes.pickle"

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, type=str, help="photos/archive/all")
# ap.add_argument("-t", "--tree", required=True, type=str, help="hash/tree.pickle")
# ap.add_argument("-a", "--hashes", required=True, type=str, help="hash/hashes.pickle")
# args = vars(ap.parse_args())

imagePaths = list(paths.list_images(IMAGES))
hashes = {}
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    image = cv2.imread(imagePath)
    h = dhash(image)
    h = convert_hash(h)
    l = hashes.get(h, [])
    l.append(imagePath)
    hashes[h] = l

print("[INFO] building VP-Tree...")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

print("[INFO] serializing VP-Tree...")
f = open(TREE, "wb")
f.write(pickle.dumps(tree))
f.close()
print("[INFO] serializing hashes...")
f = open(HASHES, "wb")
f.write(pickle.dumps(hashes))
f.close()

