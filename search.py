from hashing import convert_hash
from hashing import dhash
import pickle
import time
import cv2

VP_TREE = "hash/vptree.pickle"
HASHES = "hash/hashes.pickle"
# QUERY = "photos/samples/img_3.png"
QUERY = "src/photos/img_249.png"
DISTANCE = 18


# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--tree", required=True, type=str, help="path to pre-constructed VP-Tree")
# ap.add_argument("-a", "--hashes", required=True, type=str, help="path to hashes dictionary")
# ap.add_argument("-q", "--query", required=True, type=str, help="path to input query image")
# ap.add_argument("-d", "--distance", type=int, default=10, help="maximum hamming distance")
# args = vars(ap.parse_args())

# python search.py --tree hash/vptree.pickle --hashes hash/hashes.pickle --query photos/samples/2.jpg --distance 10

print("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open(VP_TREE, "rb").read())
hashes = pickle.loads(open(HASHES, "rb").read())
image = cv2.imread(QUERY)
cv2.imshow("Query", image)
queryHash = dhash(image)
queryHash = convert_hash(queryHash)

print("[INFO] performing search...")
start = time.time()
results = tree.get_all_in_range(queryHash, DISTANCE)
results = sorted(results)
end = time.time()
print("[INFO] search took {} seconds".format(end - start))

for d, h in results:
    print("-------------------------------------------------------------")
    resultPaths = hashes.get(h, [])
    print("[INFO] {} total image(s) with d: {}, h: {}".format(len(resultPaths), d, h))
    for resultPath in resultPaths:
        print(resultPath)
        result = cv2.imread(resultPath)
        cv2.waitKey(10)
