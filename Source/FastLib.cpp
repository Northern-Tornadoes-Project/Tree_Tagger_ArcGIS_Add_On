/* Fast Library for improving performance of python code with c++ in automated tree tagging software
*  @author Daniel Butt, NTP 2022
*  @date Aug 9, 2022
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>

//float constant of PI
#define PI       3.14159265f


//displays a progressbar for a for loop
//i = current index of loop
//total = total iterations of loop
void updateProgressBar(int i, int total){
    //assumes index starts at 1
    ++i;
    int barLength = 50;
    float progress = (float)i / (float)total;
    int pos = (int)(progress * (float)barLength);

    std::cout << "|" << std::string(pos, (char)(219)) << std::string(barLength - pos, ' ') << "| " << i << "/" << total <<"\r";
}

//shorthand for calculating the distance squared, it is faster to only calculate the distance squared than the distance and 
//commonly the true distance is not needed for comparisons (For example is dist A > dist B is equivalent to (dist A)^2 > (dist B)^2)
//x = difference in x coords
//y = difference in y coords
inline float distSquared(float x, float y){
    return x*x + y*y;
}

//finds the pair of points between two lines (one point belonging to each line) with the longest distance apart
//returns a new line between the longest point pair
inline std::array<float, 4> longestLinesPointsPair(const float p1[2], const float p2[2], const float p3[2], const float p4[2]){

    //distance squared between each possible pair
    const float A = distSquared(p1[0] - p3[0], p1[1] - p3[1]);
    const float B = distSquared(p1[0] - p4[0], p1[1] - p4[1]);
    const float C = distSquared(p2[0] - p3[0], p2[1] - p3[1]);
    const float D = distSquared(p2[0] - p4[0], p2[1] - p4[1]);

    const float maxDist = std::max(A, std::max(B, std::max(C, D)));

    //new line to return
    std::array<float, 4> l = maxDist == A ? (std::array<float, 4>{p1[0], p1[1], p3[0], p3[1]}) : 
                            (maxDist == B ? std::array<float, 4>{p1[0], p1[1], p4[0], p4[1]} : 
                            (maxDist == C ? std::array<float, 4>{p2[0], p2[1], p3[0], p3[1]} : 
                                            std::array<float, 4>{p2[0], p2[1], p4[0], p4[1]}));

    //ensure the second point of the line has the larger x coord
    //l = l[0] > l[2] ? std::array<float, 4>{l[2], l[3], l[0], l[1]} : std::array<float, 4>{l[0], l[1], l[2], l[3]};

    return l;
}

inline float distToSegmentSquared(std::array<float, 4> l, const float p[2]){

    const float l2 = distSquared(l[0] - l[2], l[1] - l[3]);
    if (l2 == 0) return distSquared(l[0] - p[0], l[1] - p[1]);

    float t = ((p[0] - l[0]) * (l[2] - l[0]) + (p[1] - l[1]) * (l[3] - l[1])) / l2;
    t = std::max(0.0f, std::min(1.0f, t));

    return distSquared(p[0] - (l[0] + t * (l[2] - l[0])), p[1] -  (l[1] + t * (l[3] - l[1])));
}


//Function for joining a set of lines from the output of a edge based line detection algorithm (openCV fast line detector).
//Joins the lines based on having similar angles, the distance from the endpoint of one line to another line segment being 
//less than a threshold, and/or, the distance and angle between two conecting endpoint (first to last or last to first) 
//being again within thresholds.
//For performance, rather than checking every line against every other line, the lines are first sorted into grid squares of 
//size 256x256 and only compared to lines in or around the same grid square.
//
//lines = list of lines and their corresponding endpoints
//x = number of horizontal grid squares (ceil(image width / 256))
//y = number of vertical grid squares (ceil(image height / 256))
//angleThreshold = max angle difference between two lines to be considered as potentially the same line
//directMergeAngleThreshold = half the search arc angle for 'directly' merging the lines
//directMergeThreshold = search arc radius for 'directly' merging the lines
//distThreshold = max distance between an endpoint and oposite line segment to be considered the same line
//minLineLength = after all lines are joined, any line smaller than this value is removed
std::vector<std::array<float, 4>> joinLines(std::vector<std::array<float, 4>> lines, const int x, const int y, const float angleThreshold, const float directMergeAngleThreshold, float directMergeThreshold, float distThreshold, float minLineLength) {

    //precalculate the angles of each line for performance
    std::vector<float> lineAngles;
    lineAngles.reserve(lines.size());

    //2d grid of vectors to store which lines are in each grid
    std::vector<std::vector<std::vector<int>>> lineGrid;

    for (int i = 0; i < y; i++) {
        lineGrid.push_back({});
        for (int j = 0; j < x; j++) {
            lineGrid.at(i).push_back({});
        }
    }

    //precalculations for performance reasons
    for (int i = 0; i < lines.size(); i++) {
        auto line = lines.at(i);

        //ensure the second point of the line has the larger x coord
        //this makes sure the angle of the line is [-90, 90] degrees
        //and helps when comparing line angles to each other
        if (line[0] > line[2]) {
            lines.at(i) = { line[2], line[3], line[0], line[1] };
        }
        //precalculate line angles
        float angle = atanf((line[3] - line[1]) / (line[2] - line[0] + 0.0001f));

        angle = fabs(angle) == 0 ? 0.000001f : angle;

        lineAngles.push_back(angle);

        //add line to all grid squares the endpoint are inside
        int a = (int)line[1] / 256;
        int b = (int)line[0] / 256;
        int c = (int)line[3] / 256;
        int d = (int)line[2] / 256;

        lineGrid.at(a).at(b).push_back(i);

        if (a != c || b != d) lineGrid.at(c).at(d).push_back(i);
    }

    //thresholds
    /*const float angleThreshold = 0.2f;
    const float directMergeAngleThreshold = 0.1f;*/
    directMergeThreshold = directMergeThreshold * directMergeThreshold; // 18
    distThreshold = distThreshold * distThreshold; //8.5
    minLineLength = minLineLength * minLineLength; //12

    //foreach grid square
    for (int grid_row_idx = 0; grid_row_idx < y; grid_row_idx++) {
        //update progressbar for each row
        updateProgressBar(grid_row_idx, y);
        for (int grid_col_idx = 0; grid_col_idx < x; grid_col_idx++) {

            //create a set (no duplicates) of all lines in or around the current grid square (3x3 grid)
            std::unordered_set<int> lineIndicies;

            for (int i = std::max(0, grid_row_idx - 1); i < std::min(y, grid_row_idx + 2); i++) {
                for (int j = std::max(0, grid_col_idx - 1); j < std::min(x, grid_col_idx + 2); j++) {
                    
                    for (auto x : lineGrid.at(i).at(j)) {
                            lineIndicies.insert(x);
                    }
                }
            }

            //turn set into a vector for easy iteration
            std::vector<int> lineIndiciesVec(lineIndicies.begin(), lineIndicies.end());

            //current amount of lines in current grid square (will chance as lines are joined)
            int sizei = lineGrid.at(grid_row_idx).at(grid_col_idx).size();

            //for each line in current grid square
            for (int i = 0; i < sizei; i++) {

                //get first line idx
                int idxi = lineGrid.at(grid_row_idx).at(grid_col_idx).at(i);

                //get first line
                auto& li = lines.at(idxi);

                //if line is already joined
                if (li[0] < 0) continue;

                //get line angle
                float liAngle = lineAngles.at(idxi);

                //current amount of lines in set of all lines in and around current grid square (will chance as lines are joined)
                int sizej = lineIndiciesVec.size();

                //compare line to all lines in and around current grid square
                for (int j = 0; j < sizej; j++) {
                    //don't compare the same line against itself
                    int idxj = lineIndiciesVec.at(j);
                    if (idxi == idxj) continue;

                    //get second line
                    auto& lj = lines.at(idxj);

                    //if second line is already joined
                    if (lj[0] < 0) continue;

                    //get second line angle
                    const float ljAngle = lineAngles.at(idxj);

                    //calculate difference in angles
                    float slopeDelta = liAngle - ljAngle;

                    //make sure the difference is calculated correctly (for example a line with angle 89 degrees looks similar to -89 degrees)
                    slopeDelta = std::min(fabs(slopeDelta), fabs(slopeDelta - PI * liAngle / fabs(liAngle)));

                    //if the lines don't meet the angle threshold
                    if (slopeDelta > angleThreshold) continue;

                    //if the distance and angle between two conecting endpoint (first to last or last to first) 
                    //is within a distance and angle threshold.
                    bool directMerge = false;

                    //first to last if distance is less than threshold
                    if(distSquared(li[0] - lj[2], li[1] - lj[3]) <= directMergeThreshold){

                        //get relative angle between end points
                        float relativePointsAngle = atan2f(lj[3] - li[1], lj[2] - li[0]);

                        //correct relative angle [-180, 180] -> [0, 360]
                        relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                        //correct precalculated line angle [-90, 90] - > [0, 360]
                        float directionLineAngle = PI + lineAngles.at(idxi);

                        //compare if angle is less than threshold
                        if(fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold){
                            directMerge = true;
                        }
                    }
                    //last to first if distance is less than threshold
                    if (!directMerge && distSquared(lj[1] - li[3], lj[0] - li[2]) <= directMergeThreshold){

                        //get relative angle between end points
                        float relativePointsAngle = atan2f(lj[1] - li[3], lj[0] - li[2]);

                        //correct relative angle [-180, 180] -> [0, 360]
                        relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                        //correct precalculated line angle ([-90, 90] +/- 180) - > [0, 360]
                        float directionLineAngle = lineAngles.at(idxi) < 0 ? 2*PI + lineAngles.at(idxi) : lineAngles.at(idxi);

                        //compare if angle is less than threshold
                        if(fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold){
                            directMerge = true;
                        }
                    }

                    //for readability
                    const float p1[2] = { lj[0], lj[1] };
                    const float p2[2] = { lj[2], lj[3] };
                    const float p3[2] = { li[0], li[1] };
                    const float p4[2] = { li[2], li[3] };

                    //if direct merge is true or the distance from the endpoint of one line to another line segment being 
                    //less than a threshold
                    if (directMerge ||
                        distToSegmentSquared(li, p1) <= distThreshold ||
                        distToSegmentSquared(li, p2) <= distThreshold ||
                        distToSegmentSquared(lj, p3) <= distThreshold ||
                        distToSegmentSquared(lj, p4) <= distThreshold) {

                        //get longest line pair
                        std::array<float, 4> LongestLine = longestLinesPointsPair(p1, p2, p3, p4);

                        //calculate the squared length of each lines
                        const float liLength = distSquared(li[2] - li[0], li[3] - li[1]);
                        const float ljLength = distSquared(lj[2] - lj[0], lj[3] - lj[1]);

                        //create a weighting based on the comparative lengths of the lines (basically weigth longers lines as being quadratically more important)
                        const float liScale = liLength / (liLength + ljLength);
                        const float ljScale = ljLength / (liLength + ljLength);
               
                        //calculated the weighted average angle of the two lines
                        float averageAngle = fabs(liAngle - ljAngle) < fabs(liAngle - ljAngle - PI * liAngle / fabs(liAngle)) ? liAngle * liScale + ljAngle * ljScale : (liAngle - PI * liAngle / fabs(liAngle)) * liScale + ljAngle * ljScale;

                        //correct the average angle if its not between [-pi/2, pi/2]
                        averageAngle = fabs(averageAngle) > PI / 2.0f ? averageAngle - PI * averageAngle / fabs(averageAngle) : averageAngle;

                        //calculate mid point of longest lines points pair
                        const float mid[2] = { (LongestLine[2] + LongestLine[0]) / 2.0f, (LongestLine[3] + LongestLine[1]) / 2.0f };

                        //get half the length of the longest lines points pair
                        const float newLineRadius = sqrtf(distSquared(LongestLine[2] - LongestLine[0], LongestLine[3] - LongestLine[1])) / 2.0f;

                        //create a new line of the same length as the longest lines points pair but rotated to be at the calculated average angle
                        const float rcos = newLineRadius * cosf(averageAngle);
                        const float rsin = newLineRadius * sinf(averageAngle);

                        std::array<float, 4> newLine = { std::max(0.0f, mid[0] - rcos), std::min(y * 256.0f - 1.0f, std::max(0.0f, mid[1] - rsin)), std::min(x * 256.0f - 1.0f, mid[0] + rcos), std::max(0.0f ,std::min(y * 256.0f - 1.0f, mid[1] + rsin)) };

                        /*if (newLine[0] > newLine[2]) {
                            newLine = { newLine[2], newLine[3], newLine[0], newLine[1] };
                        }*/

                        //get new index
                        const int newIdx = lines.size();
                        //add new line the list of lines
                        lines.push_back(newLine);
                        //calculate angle of new line
                        float angle = atanf((newLine[3] - newLine[1]) / (newLine[2] - newLine[0] + 0.0001f));

                        angle = fabs(angle) == 0 ? 0.000001f : angle;

                        lineAngles.push_back(angle);

                        //add line to grid
                        lineGrid.at(grid_row_idx).at(grid_col_idx).push_back(newIdx);
                        //increase for loop counter
                        sizei++;

                        //add line to set of lines in or around current grid square
                        lineIndiciesVec.push_back(newIdx);

                        //add line to any other grid squares it might also be in
                        int a = (int)newLine[1] / 256;
                        int b = (int)newLine[0] / 256;
                        int c = (int)newLine[3] / 256;
                        int d = (int)newLine[2] / 256;

                        if (a != grid_row_idx || b != grid_col_idx) {
                            lineGrid.at(a).at(b).push_back(newIdx);
                            if ((c != a && c != grid_row_idx) || (d != b && d != grid_col_idx)) {
                                lineGrid.at(c).at(d).push_back(newIdx);
                            }
                        }
                        else if (c != grid_row_idx || d != grid_col_idx) {
                            lineGrid.at(c).at(d).push_back(newIdx);
                        }

                        //set x coord of first point of each now joined line to signify that the lines were joined
                        lines.at(idxi)[0] = -1.0f;
                        lines.at(idxj)[0] = -1.0f;
                        //no need to check the first line any more
                        break;
                    }

                }
            }

        }
    }
    //adds new line after progress bar
    std::cout << std::endl;

    std::vector<std::array<float, 4>> newLines;
    newLines.reserve((int)(lines.size() * 0.5));

    for (auto& l : lines) {
        if (l[0] >= 0 && distSquared(l[2] - l[0], l[3] - l[1]) >= minLineLength) {
            newLines.push_back(l);
        }
    }

    return newLines;

    //int i = lines.size() - 1;

    ////loop through all lines and remove small lines / lines that were joined
    ////looping through backwards is faster under the circumstances
    //
    //while (i >= 0) {
    //    auto& l = lines.at(i);

    //    if (l[0] < 0 || distSquared(l[2] - l[0], l[3] - l[1]) < minLineLength) {
    //        lines.erase(lines.begin() + i);
    //    }
    //    i--;
    //}

    //return lines;
}

float median(std::vector<float> vec){

    const int n = vec.size();

    nth_element(vec.begin(), vec.begin() + n / 2, vec.end());

    if (n % 2 == 0) {

        nth_element(vec.begin(), vec.begin() + (n - 1) / 2, vec.end());

        return (vec.at((n - 1) / 2) + vec.at(n / 2)) / 2.0f;
    }

    else {

        return vec.at(n/2);
    }
}


std::vector<std::vector<std::array<float, 3>>> averageDirections(std::vector<std::array<float, 5>> lines, const float gridSize, const int x, const int y, const int extrapolationReq, const int minNumOfLines) {

    std::vector<std::vector<std::array<float, 3>>> averageDirectionsGrid;
    for (int i = 0; i < y; i++) {
        averageDirectionsGrid.push_back({});
        for (int j = 0; j < x; j++) {
            averageDirectionsGrid.at(i).push_back({1e10f, 1e10f, -1.0f});
        }
    }

    //2d grid of vectors to store which lines are in each grid
    std::vector<std::vector<std::vector<int>>> lineGrid;

    for (int i = 0; i < y; i++) {
        lineGrid.push_back({});
        for (int j = 0; j < x; j++) {
            lineGrid.at(i).push_back({});
        }
    }

    //add lines to grid based on mid points
    for (int i = 0; i < lines.size(); i++) {
        const auto line = lines.at(i);
        const float mid[2] = { (line[0] + line[2]) / 2.0f, (line[1] + line[3]) / 2.0f };

        lineGrid.at(floorf(mid[1] / gridSize)).at(floorf(mid[0] / gridSize)).push_back(i);
    }

    //for each grid square, average the direction by averaging the unit vectors of the lines weighted by the machine learning model's confidence
    for (int gridRowIdx = 0; gridRowIdx < y; gridRowIdx++) {
        for (int gridColIdx = 0; gridColIdx < x; gridColIdx++) {
            float sum[2] = { 0.0f, 0.0f };
            float total = 0.0f;

            const auto gridSquare = lineGrid.at(gridRowIdx).at(gridColIdx);
            const int numOfLines = gridSquare.size();

            if (numOfLines < minNumOfLines) continue;

            for (auto lineIdx : gridSquare) {
                const auto line = lines.at(lineIdx);

                const float lineVector[2] = {line[2] - line[0], line[3] - line[1]};

                //compute unit vector and add to sum
                const float magnitude = sqrtf(distSquared(lineVector[0], lineVector[1]));

                sum[0] += (lineVector[0] / magnitude)*line[4];
                sum[1] += (lineVector[1] / magnitude)*line[4];
                total += line[4];
            }

            
            const float averageVec[2] = {sum[0] / total, sum[1] / total};
            const float averageMagnitude = sqrtf(distSquared(averageVec[0], averageVec[1]));

            averageDirectionsGrid.at(gridRowIdx).at(gridColIdx) = { averageVec[0] / averageMagnitude, averageVec[1] / averageMagnitude, averageMagnitude };

            /*if (averageMagnitude > 0.6666666f) {
                averageDirectionsGrid.at(gridRowIdx).at(gridColIdx) = { averageVec[0] / averageMagnitude, averageVec[1] / averageMagnitude, 4.0f };
            }
            else if (averageMagnitude > 0.3333333f) {
                averageDirectionsGrid.at(gridRowIdx).at(gridColIdx) = { averageVec[0] / averageMagnitude, averageVec[1] / averageMagnitude, 3.0f };
            }
            else {
                averageDirectionsGrid.at(gridRowIdx).at(gridColIdx) = { averageVec[0] / averageMagnitude, averageVec[1] / averageMagnitude, 2.0f };
            }*/
        }
 
    }

    //int minNumOfSurrounding = 8;

    ////extrapolation
    //while (true) {
    //    bool found = false;

    //    for (int gridRowIdx = 0; gridRowIdx < y; gridRowIdx++) {
    //        for (int gridColIdx = 0; gridColIdx < x; gridColIdx++) {

    //            if (averageDirectionsGrid.at(gridRowIdx).at(gridColIdx)[0] < 1e9f) continue;

    //            std::vector<float> xDirections;
    //            std::vector<float> yDirections;
    //            int total = 0;

    //            for (int i = std::max(0, gridRowIdx - 1); i < std::min(y, gridRowIdx + 2); i++) {
    //                for (int j = std::max(0, gridColIdx - 1); j < std::min(x, gridColIdx + 2); j++) {
    //                    const auto direction = averageDirectionsGrid.at(i).at(j);

    //                    if (direction[0] < 1e9f) {
    //                        total++;
    //                        xDirections.push_back(direction[0]);
    //                        yDirections.push_back(direction[1]);
    //                    }
    //                }
    //            }
    //            if (total >= minNumOfSurrounding) {
    //                found = true;

    //                const float medianVec[2] = { median(xDirections), median(yDirections) };
    //                const float medianMagnitude = sqrtf(distSquared(medianVec[0], medianVec[1]));

    //                averageDirectionsGrid.at(gridRowIdx).at(gridColIdx) = {medianVec[0] / medianMagnitude, medianVec[1] / medianMagnitude, 0.0f};
    //            }

    //        }
    //    }

    //    if (!found && minNumOfSurrounding == extrapolationReq) break;

    //    if (minNumOfSurrounding > extrapolationReq) minNumOfSurrounding--;
    //}

    return averageDirectionsGrid;
}


//pybind 11 boilerplate for compiling to python binary
PYBIND11_MODULE(FastLib, handle){
    handle.doc() = "Fast Library for improving performance of python code with c++ in automated tree tagging software";
    handle.def("joinLines", &joinLines);
    handle.def("averageDirections", &averageDirections);
    
}