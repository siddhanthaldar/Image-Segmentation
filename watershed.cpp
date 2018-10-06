#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<limits.h>

using namespace std;
using namespace cv;

void bfs(int** mark, int** Cn, int m, int n, int set_no, Mat img)
{
	for(int i = m - 1; i <= m + 1; i++)
		for(int j = n - 1; j <= n + 1; j++)
		{
			if(i < 0 || j < 0 || i >= img.rows || j >= img.cols) continue;
			if(i == j) continue;
			if(Cn[i][j] != 0 && mark[i][j] == 0)
			{
				Cn[i][j] = set_no;
				mark[i][j] = 1;
				bfs(mark, Cn, i, j, set_no, img);
			}
		}
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		cout<<"Please provide file name\n";
		return 0;
	}
	
	//Input image and Initialize Q
	Mat img = imread(argv[1], 0);

	Mat boundary(img.rows, img.cols, CV_8UC1, Scalar(0));
	// Sobel(img, img)
	int set_no = 2;
	int** Q = new int*[img.rows];
	for(int i = 0; i < img.rows; i++) Q[i] = new int[img.cols]();

	for(int level = 0; level < 256; level++)
	{
		
		int **Cn = new int*[img.rows];
		for(int i = 0; i < img.rows; i++) Cn[i] = new int[img.cols]();

		for(int i = 0; i < img.rows; i++)
			for(int j = 0; j < img.cols; j++)
				if(img.at<uchar>(i, j) == level)
					Cn[i][j] = 1;

		int** mark = new int*[img.rows];
		for(int i = 0; i < img.rows; i++) mark[i] = new int[img.cols]();
		
		int set_no_loc = 2;
		for(int i = 0; i < img.rows; i++)
			for(int j = 0; j < img.cols; j++)
				if(Cn[i][j] != 0 && mark[i][j] == 0)
				{
					mark[i][j] = 1;
					Cn[i][j] = set_no_loc;
					bfs(mark, Cn, i, j, set_no_loc, img);
					set_no_loc++;
				}

		for(int i = 0; i < img.rows; i++)
			delete mark[i];
		delete mark;

		vector<int> *neighbour = new vector<int>[set_no_loc];

		for(int i = 0; i < img.rows; i++)
			for(int j = 0; j < img.cols; j++)
			{
				if(Cn[i][j] == 0) continue;
				for(int m = i - 1; m <= i + 1; m++)
					for(int n = j - 1; n <= j + 1; n++)
					{
						if(Q[m][n] == 0) continue;
						neighbour[Cn[i][j]].push_back(Q[m][n]);
					}
			}
		
		for(int set = 2; set < set_no_loc; set++)
		{
			//Case 1 : No Neighbours.
			if(neighbour[set].size() == 0)
			{
				for(int i = 0; i < img.rows; i++)
					for(int j = 0; j < img.cols; j++)
						if(Cn[i][j] == set)
							Q[i][j] = set_no;
				set_no++;
				continue;
			}
			
			//Case 2 : 1 Neighbour
			int flag = 0;
			for(int i = 1; i < neighbour[set].size(); i++)
				if(neighbour[set][0] != neighbour[set][i])
				{
					flag = 1;
					break;
				}
			if(!flag)
			{
				for(int i = 0; i < img.rows; i++)
					for(int j = 0; j < img.cols; j++)
						if(Cn[i][j] == set)
							Q[i][j] = neighbour[set][0];
			}

			//Case 3 : More than 1 Neighbour
			else
			{
				//Finding unique intersections
				vector<int> unique;
				for(int i = 2; i < set_no; i++)
					for(int j = 0; j < neighbour[set].size(); j++)
						if(i == neighbour[set][j])
						{
							unique.push_back(i);
							break;
						}

				//Finding union of final sets
				int** T = new int*[img.rows];
				for(int i = 0; i < img.rows; i++) T[i] = new int[img.cols]();

				int min = INT_MAX;
				for(int i = 0; i < unique.size(); i++) if(min > unique[i]) min = unique[i];

				for(int i = 0; i < img.rows; i++)
					for(int j = 0; j < img.cols; j++)
					{
						if(Q[i][j] == 0) continue;
						int f = 0;
						for(int k = 0; k < unique.size(); k++)
							if(Q[i][j] == unique[k])
								f = 1;
						if(f) T[i][j] = 1;

						if(Cn[i][j] == set)
							T[i][j] = 1;
					}

				int** D = new int*[img.rows];
				for(int i = 0; i < img.rows; i++) D[i] = new int[img.cols]();

				for(int i = 0; i < img.rows; i++)
					for(int j = 0; j < img.cols; j++)
						D[i][j] = Q[i][j];

				int T_visited = 0;
				for(int i = 0; i < img.rows; i++)
					for(int j = 0; j < img.cols; j++)
						if(T[i][j] == 1) T_visited++;

				while(1)
				{
					int** D_1 = new int*[img.rows];
					for(int i = 0; i < img.rows; i++) D_1[i] = new int[img.cols]();

					for(int i = 0; i < img.rows; i++)
						for(int j = 0; j < img.cols; j++)
							D_1[i][j] = D[i][j];

					for(int i = 0; i < img.rows; i++)
						for(int j = 0; j < img.cols; j++)
						{

							if(T[i][j] == 0) continue;
							if(D[i][j] != 0)
							{
								if(T[i][j] != 2)
								{
									T[i][j] = 2;
									T_visited--;
								}
								continue;
							}
							for(int m = i - 1; m  <= i + 1; m++)
								for(int n = j - 1; n <= j + 1; n++)
								{
									if(D[m][n] == 0) continue;
									if(T[i][j] != 2)
									{
										T_visited--;
										T[i][j] = 2;
									}
									if(D_1[i][j] == 0) D_1[i][j] = D[m][n];
									else
									{
										if(D[m][n] == D_1[i][j]) continue;
										boundary.at<uchar>(i, j) = 255;
										break;
									}

								}
						}

					for(int i = 0; i < img.rows; i++)
						for(int j = 0; j < img.cols; j++)
							D[i][j] = D_1[i][j];

					if(!T_visited) break;
				}

			}

		}
	}

	imshow("boundary", boundary);
	waitKey(0);

}