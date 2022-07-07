/* Fast Library for improving performance of python code with c++ in automated tree tagging software
*  @author Daniel Butt, NTP 2022
*  @date June 24, 2022
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <array>
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
    l = l[0] > l[2] ? std::array<float, 4>{l[2], l[3], l[0], l[1]} : std::array<float, 4>{l[0], l[1], l[2], l[3]};

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
std::vector<std::array<float, 4>> joinLines(std::vector<std::array<float, 4>> lines, const int x, const int y) {

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
    const float angleThreshold = 0.2f;
    const float directMergeAngleThreshold = 0.10f;
    const float directMergeThreshold = 256.0f;
    const float distThreshold = 72.0f;
    const float minLineLength = 15.0f * 15.0f;

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
                    slopeDelta = std::min(abs(slopeDelta), abs(slopeDelta - PI * liAngle / fabs(liAngle)));

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
                    else if (distSquared(lj[1] - li[3], lj[0] - li[2]) <= directMergeThreshold){

                        //get relative angle between end points
                        float relativePointsAngle = atan2f(lj[1] - li[3], lj[0] - li[2]);

                        //correct relative angle [-180, 180] -> [0, 360]
                        relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                        //correct precalculated line angle ([-90, 90] +/- 180) - > [0, 360]
                        float directionLineAngle = lineAngles.at(idxi) < 0 ? 2*PI + lineAngles.at(idxi) : lineAngles.at(idxi);

                        //compare if angle is less than threshold
                        if(fabs(relativePointsAngle - lineAngles.at(idxi)) <= directMergeAngleThreshold){
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

                        const float liLength = distSquared(li[2] - li[0], li[3] - li[1]);
                        const float ljLength = distSquared(lj[2] - lj[0], lj[3] - lj[1]);

                        const float liScale = liLength / (liLength + ljLength);
                        const float ljScale = ljLength / (liLength + ljLength);
               

                        float averageAngle = fabs(liAngle - ljAngle) < fabs(liAngle - ljAngle - PI * liAngle / fabs(liAngle)) ? liAngle * liScale + ljAngle * ljScale : (liAngle - PI * liAngle / fabs(liAngle)) * liScale + ljAngle * ljScale;

                        /*const float angleTotal = liAngle + ljAngle;
                        float averageAngle = std::max(fabs(angleTotal), fabs(angleTotal - PI * liAngle / fabs(liAngle))) / 2.0;*/
                        averageAngle = fabs(averageAngle) > PI / 2.0f ? averageAngle - PI * averageAngle / fabs(averageAngle) : averageAngle;

                        const float mid[2] = { (LongestLine[2] + LongestLine[0]) / 2.0f, (LongestLine[3] + LongestLine[1]) / 2.0f };

                        /*const float longestLineAngle = atanf((LongestLine[3] - LongestLine[1]) / (LongestLine[2] - LongestLine[0] + 0.0001f));
                        const float longestLineRadius = sqrtf(distSquared(LongestLine[3] - LongestLine[1], LongestLine[2] - LongestLine[0])) / 2;

                        const float newLineRadius = fabs(longestLineRadius * std::cosf(std::min(averageAngle - longestLineAngle, averageAngle - longestLineAngle - PI * averageAngle / fabs(averageAngle))));*/

                        const float newLineRadius = sqrtf(distSquared(LongestLine[2] - LongestLine[0], LongestLine[3] - LongestLine[1])) / 2.0f;

                        const float rcos = newLineRadius * cosf(averageAngle);
                        const float rsin = newLineRadius * sinf(averageAngle);

                        std::array<float, 4> newLine = { std::max(0.0f, mid[0] - rcos), std::min(y * 256.0f - 1.0f, std::max(0.0f, mid[1] - rsin)), std::min(x * 256.0f - 1.0f, mid[0] + rcos), std::max(0.0f ,std::min(y * 256.0f - 1.0f, mid[1] + rsin)) };

                        if (newLine[0] == 0 && newLine[1] == 0 && newLine[2] == x * 256.0f - 1.0f && newLine[3] == y * 256.0f - 1.0f) {

                            std::cout << newLineRadius << ", " << averageAngle << ", " << liAngle << ", " << ljAngle << std::endl;
                        }


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

                        //set x coord of first point of each now joined line to signify that the lines wee joined
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

    int i = lines.size() - 1;

    //loop through all lines and remove small lines / lines that were joined
    //looping through backwards is faster under the circumstances
    while (i >= 0) {
        auto& l = lines.at(i);

        if (l[0] < 0 || distSquared(l[2] - l[0], l[3] - l[1]) < minLineLength) {
            lines.erase(lines.begin() + i);
        }
        i--;
    }

    return lines;
}


//pybind 11 boilerplate for compiling to python binary
PYBIND11_MODULE(FastLib, handle){
    handle.doc() = "Fast Library for improving performance of python code with c++ in automated tree tagging software";
    handle.def("joinLines", &joinLines);
    
}

//ignore

// inline float distToSegmentSquared2(std::array<float, 4> l, const float p[2]){
//     const float A = p[0] - l[0];
//     const float B = p[1] - l[1];
//     const float C = l[2] - l[0];
//     const float D = l[3] - l[1];

//     const float dot = A * C + B * D;
//     const float len_sq = distSquared(C, D);
//     const float param = (len_sq != 0) ? dot / len_sq : -1.0f;

//     float xx, yy;

//     if (param < 0 ) {
//         xx = l[0];
//         yy = l[1];
//     }
//     else if (param > 1) {
//         xx = l[2];
//         yy = l[3];
//     }
//     else {
//         xx = l[0] + param * C;
//         yy = l[1] + param * D;
//     }

//     const float dx = p[0] - xx;
//     const float dy = p[1] - yy;

//     return distSquared(dx, dy);
// }

// inline float distPointLine3d(std::array<float, 3> p, std::array<float, 3> l){

//     float w = sqrtf(distSquared(l[0], l[1]));
//     std::array<float, 3> k = {l[0] / w, l[1] / w, l[2] / w};

//     return p[0] * k[0] + p[1] * k[1] + p[2] * k[2];
// }

// inline std::array<float, 3> cross(std::array<float, 3> a, std::array<float, 3> b){
//     return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
// }

// bool fineMerge(std::array<float, 4> li, std::array<float, 4> lj){

//     std::array<float, 3> o = {(lj[0] + lj[2]) * 0.5f, (lj[1] + lj[3]) * 0.5f, 1.0f};
//     std::array<float, 3> p1 = {li[0], li[1], 1.0f};
//     std::array<float, 3> p2 = {li[2], li[3], 1.0f};

//     std::array<float, 3> cl = cross(p1, p2);

//     const float midix = (li[0] + li[2]) * 0.5f;
//     const float midiy = (li[1] + li[3]) * 0.5f;

//     //TODO
//     float segilen = sqrtf(distSquared(li[2]-li[0], li[3]-li[1]));
//     float segjlen = sqrtf(distSquared(lj[2]-lj[0], lj[3]-lj[1]));
//     float middist = sqrtf(distSquared(midix - o[0], midiy - o[1]));

//     float dist = fabs(distPointLine3d(o, cl));

//     if(dist <= 1.414213562f * 2.0f && middist <= segilen * 0.5f + segjlen * 0.5f + 20.0f) return true;

//     return false;
// }