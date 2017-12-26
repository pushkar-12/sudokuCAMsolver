import cv2
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn import svm
from sklearn import datasets, neighbors, linear_model
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC




def draw_str(dst, x, y, s):
    """
    Draw a string with a dark contour
    """
    cv2.putText(dst, s, (x + 1, y + 1),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)




def cmp_heightitem(item):
    _, _, _, hitem = cv2.boundingRect(item)
    return -hitem


def cmp_widthitem(item):
    """used for sorting by width"""
    _, _, witem, _ = cv2.boundingRect(item)

    return - witem


def sort_grid_points(points):
    """
    Given a flat list of points (x, y), this function returns the list of
    points sorted from top to bottom, then groupwise from left to right.

    We assume that the points are nearly equidistant and have the form of a
    square.
    """
    w, _ = points.shape
    sqrt_w = int(np.sqrt(w))
    # sort by y
    points = points[np.argsort(points[:, 1])]
    # put the points in groups (rows)
    points = np.reshape(points, (sqrt_w, sqrt_w, 2))
    # sort rows by x
    points = np.vstack([row[np.argsort(row[:, 0])] for row in points])
    # undo shape transformation
    points = np.reshape(points, (w, 1, 2))
    return points


def process(frame):

    #
    # 1. preprocessing
    #
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        src=gray, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

    blurred = cv2.medianBlur(binary, ksize=3)

    #
    # 2. try to find the sudoku
    #
    image,contours, hierarchy = cv2.findContours(image=cv2.bitwise_not(blurred),
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)



    sudoku_area = 0
    sudoku_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (0.7 < float(w) / h < 1.3     # aspect ratio
                and area > 150 * 150     # minimal area
                and area > sudoku_area   # biggest area on screen
                and area > .5 * w * h):  # fills bounding rect
            sudoku_area = area
            sudoku_contour = cnt

    #blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(blurred, [sudoku_contour], -1, (255,0,0), -1)




    #
    # 3. separate sudoku from background
    #
    if sudoku_contour is not None:

        # approximate the contour with connected lines
        perimeter = cv2.arcLength(curve=sudoku_contour, closed=True)
        approx = cv2.approxPolyDP(curve=sudoku_contour,
                                  epsilon=0.1 * perimeter,
                                  closed=True)

        if len(approx) == 4:
            # successfully approximated
            # we now transform the sudoku to a fixed size 450x450
            # plus 50 pixel border and remove the background

            # create empty mask image
            mask = np.zeros(gray.shape, np.uint8)
            # fill a the sudoku-contour with white
            #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(mask, [sudoku_contour], 0, 255, -1)

            # invert the mask
            mask_inv = cv2.bitwise_not(mask)
            # the blurred picture is already thresholded so this step shows
            # only the black areas in the sudoku
            separated = cv2.bitwise_or(mask_inv, blurred)



            # create a perspective transformation matrix. "square" defines the
            # target dimensions (450x450). The image we warp "separated" in
            # has bigger dimensions than that (550x550) to assure that no
            # pixels are cut off accidentially on twisted images
            square = np.float32([[50, 50], [500, 50], [50, 500], [500, 500]])
            approx = np.float32([i[0] for i in approx])  # api needs conversion
            # sort the approx points to match the points defined in square
            approx = sort_grid_points(approx)


            m = cv2.getPerspectiveTransform(approx, square)
            transformed = cv2.warpPerspective(separated, m, (550, 550))



            #
            # 4. get crossing points to determine grid buckling
            #

            # 4.1 vertical lines
            #

            # sobel x-axis
            sobel_x = cv2.Sobel(transformed, ddepth=-1, dx=1, dy=0)



            # closing x-axis
            kernel_x = np.array([[1]] * 20, dtype='uint8')  # vertical kernel
            dilated_x = cv2.dilate(sobel_x, kernel_x)
            closed_x = cv2.erode(dilated_x, kernel_x)
            _, threshed_x = cv2.threshold(closed_x, thresh=250, maxval=255,
                                          type=cv2.THRESH_BINARY)



            # generate mask for x
            image1,contours1, hierarchy1 = cv2.findContours(image=threshed_x,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)


            # sort contours by height
            sorted_contours1 = sorted(contours1, key=cmp_heightitem)

            # fill biggest 10 contours on mask (white)
            mask_x = np.zeros(transformed.shape, np.uint8)
            cv2.drawContours(mask_x, sorted_contours1[:10], -1, 255, -1)







            # 4.2 horizontal lines
            #

            # this is essentially the same procedure as for the x-axis
            # sobel y-axis
            sobel_y = cv2.Sobel(transformed, ddepth=-1, dx=0, dy=1)

            # closing y-axis
            kernel_y = np.array([[[1]] * 20], dtype='uint8')  # horizontal krnl
            dilated_y = cv2.dilate(sobel_y, kernel_y)
            closed_y = cv2.erode(dilated_y, kernel_y)
            _, threshed_y = cv2.threshold(closed_y, 250, 255,
                                          cv2.THRESH_BINARY)

            # generate mask for y
            image,contours2, _ = cv2.findContours(image=threshed_y,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours2 = sorted(contours2, key=cmp_widthitem)

            # fill biggest 10 on mask
            mask_y = np.zeros(transformed.shape, np.uint8)
            cv2.drawContours(mask_y, sorted_contours2[:10], -1, 255, -1)

            #
            # 4.3 close the grid
            #
            dilated_ver = cv2.dilate(mask_x, kernel_x)
            dilated_hor = cv2.dilate(mask_y, kernel_y)
            # now we have the single crossing points as well as the complete
            # grid
            grid = cv2.bitwise_or(dilated_hor, dilated_ver)
            crossing = cv2.bitwise_and(dilated_hor, dilated_ver)
            cv2.imshow('image',crossing)

            #
            # 5. sort crossing points
            #
            image,contours, _ = cv2.findContours(image=crossing,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            # a complete sudoku must have exactly 100 crossing points
            if len(contours) == 100:
                # take the center points of the bounding rects of the crossing
                # points. This should be precise enough, calculating the
                # moments is not necessary.
                crossing_points = np.empty(shape=(100, 2))
                for n, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = (x + .5 * w, y + .5 * h)
                    crossing_points[n] = [int(cx), int(cy)]
                sorted_cross_points = sort_grid_points(crossing_points)
                # show the numbers next to the points
                for n, p in enumerate(sorted_cross_points):
                    #print (p[0])
                    draw_str(grid, int(p[0][0]), int(p[0][1]), str(n))

                #
                # 6. Solve the sudoku
                #
                solve_sudoku_ocr(transformed, sorted_cross_points)

    cv2.drawContours(frame, [sudoku_contour], 0, 255)
    cv2.imshow('Input', frame)



def solve_sudoku_ocr(src, crossing_points):
    """
    Split the rectified sudoku image into smaller pictures of letters only.
    Then perform ocr on the letter images, create and solve the sudoku using
    the Sudoku class.
    """
    numbers = []
    # enumerate all the crossing points except the ones on the far right border
    # to get the single cells


    samples = np.empty((0, 100))
    responses = []
    cnt = 0
    for i, pos in enumerate([pos for pos in range(89) if (pos + 1) % 10 != 0]):

        # warp the perspective of the cell to match a square.
        # the target image "transformed" is slightly smaller than "square" to
        # cut off noise on the borders
        square = np.float32([[-10, -10], [40, -10], [-10, 40], [40, 40]])
        # get the corner points for the cell i
        quad = np.float32([crossing_points[pos],
                           crossing_points[pos + 1],
                           crossing_points[pos + 10],
                           crossing_points[pos + 11]])

        matrix = cv2.getPerspectiveTransform(quad, square)
        transformed = cv2.warpPerspective(src, matrix, (30, 30))
        transformed = cv2.bitwise_not(transformed)
        x=np.zeros((50,50),np.uint8)
        for i in range(10,40):
            for j in range(10,40):
                x[i][j]=transformed[i-10][j-40]
        #transformed=cv2.resize(x,(30,30),interpolation=cv2.INTER_CUBIC)
        transformed=x

        #ret, transformed = cv2.threshold(transformed, 100, 255, cv2.THRESH_BINARY)

        kernel=np.ones((3,3),np.uint8)

        transformed=cv2.dilate(transformed,kernel,iterations = 1)

        transformed = cv2.bitwise_not(transformed)
        ret, transformed = cv2.threshold(transformed, 100, 255, cv2.THRESH_BINARY)
        imgg,contours,hierarchy=cv2.findContours(transformed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        maxarea=-1
        index=-1
        maxcnt=[]
        for i in range(len(contours)):

            area=cv2.contourArea(contours[i])

            if(area>maxarea):
                maxarea=area
                index=i
                maxcnt=contours[i]
            #cv2.rectangle(transformed, (x, y), (x + w, y + h), (0, 255, 0), 2)


        contours.remove(maxcnt)
        #transformed=cv2.cvtColor(transformed,cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(transformed,contours,index,(0,255,0),3)

        try:
            maxarea=-1
            index=-1

            for i in range(len(contours)):

                area=cv2.contourArea(contours[i])

                if(area>maxarea):
                    maxarea=area
                    index=i


            x,y,w,h=cv2.boundingRect(contours[index])
            #transformed=cv2.cvtColor(transformed,cv2.COLOR_GRAY2BGR)
            #cv2.rectangle(transformed, (x, y), (x + w, y + h), (0, 255, 0), 2)

            transformed=transformed[y-2:y+h+2,x-2:x+w+2]
            transformed=cv2.resize(transformed,(10,10))
            cv2.imshow('image', transformed)

            #transformed=cv2.cvtColor(transformed,cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(0)

            if key <= 57 and key >= 48:
                responses.append((key))
                sample = transformed.reshape((1, 100))
                samples = np.append(samples, sample, 0)
            cnt += 1
            print(cnt)
            if (key == ord('q')):
                break
            cv2.destroyAllWindows()
        except:
            continue

    responses = np.array(responses, np.uint8)
    responses = responses.reshape((responses.size, 1))
    print("training complete")
    np.savetxt('generalsamples6.data', samples)
    np.savetxt('generalresponses6.data', responses)


"""Uses the main video capture device for detection"""
"""Z
cap = cv2.VideoCapture(0)
if cap.isOpened():
    while(not cv2.waitKey(1) & 0xFF == ord('q')):
        _, frame = cap.read()
        process(frame)
else:
    raise IOError('Cannot capture video device')
cap.release()
"""
frame=cv2.imread('ex.png');
process(frame)