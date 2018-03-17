import java.util.*;
import java.util.Arrays;
import java.io.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.Buffer;

public class	probGen{

	public int attr;
	public Vector data;
	public Vector<Double> target;	//saves target values
	public int inputSize;
	public int num_classes;
	public Double[] w;
	public double w0;
	public Double[][] classMeans;		//vector of class means where each mean is 4X1 matrix
	public Vector classVariance;	//vector of class variances where each variance is 4X4 matrix
	public Double[][] covariance;
	public int[] classSum;

	//test variables
	public Vector test_data;	//input points are stored here
	public Vector<Double> test_label;	//saves target test label values
	public Vector<Double> pred_label;	//computed test label values
	public int test_size;

	public probGen(int a,int c){
		attr = a;
		data = new Vector();
		target = new Vector<Double>();
		inputSize = 0;
		num_classes = c;

		w = new Double[attr];
		//w0 = new Double[attr];

		classMeans = new Double[num_classes][attr];	//declare and initialise
		for(int i=0;i<num_classes;i++)
		{
			for(int j=0;j<attr;j++)
				classMeans[i][j] = 0.0;
		}

		covariance = new Double[attr][attr];	//declare and initialise
		for(int k=0;k<attr;k++)
			{
				for(int l=0;l<attr;l++)
				{
					covariance[k][l] = 0.0;
				}

			}

		classVariance = new Vector();
		
		classSum = new int[num_classes];	//declare and initialise
		for(int i=0;i<num_classes;i++)
			classSum[i] =0;

		//intialize test variables here
		test_data = new Vector();
		test_label = new Vector<Double>();
		pred_label = new Vector<Double>();
		test_size = 0;

	}

	public void ReadFileBuffReader(){
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(
					"/home/tarunee/4-2/ML2/ML2 - Assignment 1/train.txt"));
			String line = reader.readLine();

			while (line != null) {
				
				String[] tokens=line.split(",");
				inputSize++;
				
				//temp contains (VEctor) all the 5 attributes
				//the first 4 attributes are real numbers.
				//1.variance 2.skewness 3.curtosis 4.entropy 5.class(0/1)
				
				int i=0;
				double num =0;
				Vector<Double> attributes = new Vector<Double>();	//array of attr values
				for(String key : tokens){
					num = (Double)Double.parseDouble(key);

					if(i==attr)//for the last class labelled value
					{
						break;

					}
					attributes.addElement(num);
					i++;		
				}
				if(i==4){//---->4 is same as attr
					target.addElement(num);
					//class computations
					if(num == 0.0)
					{
						classSum[0]++;
					}
					else{
						classSum[1]++;
					}

				}

				data.addElement(attributes);	//inserting each data line.
				
				// read next line
				line = reader.readLine();
			}

			
			//System.out.println(items_info);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
}

public void classCompute(){

		//class means arer alread y intialised to zero
		//in constructor itself

		for(int j=0;j<inputSize;j++)
		{
			Double[] xn = new Double[attr];
			Vector<Double> x = new Vector<Double>();
			x = (Vector)data.elementAt(j);
			x.toArray(xn);	//convert vector to array
			double tn = target.elementAt(j);

			for(int i=0;i<attr;i++)
			{
				classMeans[0][i] = classMeans[0][i] + xn[i]*(1-tn);
				classMeans[1][i] = classMeans[1][i] + xn[i]*tn;

			}
		}//loop for reading data

		//final class means
		for(int i=0;i<attr;i++)
		{
			classMeans[0][i] = (double)classMeans[0][i]/classSum[0];
			classMeans[1][i] = (double)classMeans[1][i]/classSum[1];

		}

		//compute class variance
		for(int i=0;i<num_classes;i++)
		{
			Double[] new_x ;
			Double[][] varS ;
			Double[][] totalS = new Double[attr][attr];
			for(int k=0;k<attr;k++)
			{
				for(int l=0;l<attr;l++)
				{
					totalS[k][l] = 0.0;
				}

			}

			for(int j=0;j<inputSize;j++)
			{

				Double[] xn = new Double[attr];
				Vector<Double> x = new Vector<Double>();
				x = (Vector)data.elementAt(i);
				x.toArray(xn);	//convert vector to array
				double tn = target.elementAt(j);

				new_x = new Double[attr];
				varS = new Double[attr][attr];

				if((int)tn == i)
				{
					for(int k=0;k<attr;k++)
					{	//(xn - m)
						new_x[k] = xn[k] - classMeans[i][k];
					}

					for(int k=0;k<attr;k++)
					{	//(xn - m)(xn - m)T
						for(int l=0;l<attr;l++)
						{
							varS[k][l] = new_x[k]*new_x[l];
						}

					}//calcualting variance for givrn n

					//adding final variance of a class
					//performing sigma for all n
					for(int k=0;k<attr;k++)
					{
						for(int l=0;l<attr;l++)
						{
							totalS[k][l] = totalS[k][l] + varS[k][l];
						}

					}

				}//end if that x belongs to the given class

			}//end j for each data input

			classVariance.addElement(totalS);
		}//end i for each clas


	//calculating covariance

	for(int i=0;i<num_classes;i++)
	{
		Double[][] clsvar = (Double[][])classVariance.elementAt(i);
		
		for(int k=0;k<attr;k++)
		{
			for(int l=0;l<attr;l++)
			{
				covariance[k][l] = covariance[k][l] + clsvar[k][l];
			}
		}
	}

	//final covariance matrix
	for(int k=0;k<attr;k++)
	{
		for(int l=0;l<attr;l++)
		{
			covariance[k][l] = (double)covariance[k][l]/inputSize;
		}

	}

}

public Double[] matrix_mult(Double[][] a,Double[] b, int m,int n)
{
	Double[] ans = new Double[m];

	for(int i=0;i<m;i++)
	{
		double sum = 0;
		for(int k=0;k<n;k++)
		{
			sum = sum + a[i][k]*b[k];
		}
		ans[i] = sum;

	}

	return ans;
}

public Double[] matrix_mult(Double[] b,Double[][] a, int m,int n)
{
	Double[] ans = new Double[m];

	for(int i=0;i<m;i++)
	{
		double sum = 0;
		for(int k=0;k<n;k++)
		{
			sum = sum + a[k][i]*b[k];
		}
		ans[i] = sum;

	}

	return ans;
}


//////***************//////
public void gaussian(Double a[][], int index[])
{
        int n = index.length;
        double c[] = new double[n];
        
 
 // Initialize the index
        for (int i=0; i<n; ++i) 
            index[i] = i;
 
 // Find the rescaling factors, one from each row
        for (int i=0; i<n; ++i) 
        {
            double c1 = 0;
            for (int j=0; j<n; ++j) 
            {
                double c0 = Math.abs(a[i][j]);
                if (c0 > c1) c1 = c0;
            }
            c[i] = c1;
        }
 
 // Search the pivoting element from each column
        int k = 0;
        for (int j=0; j<n-1; ++j) 
        {
            double pi1 = 0;
            for (int i=j; i<n; ++i) 
            {
                double pi0 = Math.abs(a[index[i]][j]);
                pi0 /= c[index[i]];
                if (pi0 > pi1) 
                {
                    pi1 = pi0;
                    k = i;
                }
            }
 
   // Interchange rows according to the pivoting order
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;
            for (int i=j+1; i<n; ++i) 	
            {
                double pj = a[index[i]][j]/a[index[j]][j];
 
 // Record pivoting ratios below the diagonal
                a[index[i]][j] = pj;
 
 // Modify other elements accordingly
                for (int l=j+1; l<n; ++l)
                    a[index[i]][l] -= pj*a[index[j]][l];
            }
        }

}//end of gaussian

 public Double[][] invert(Double a[][]){
        int n = a.length;
        Double x[][] = new Double[n][n];
        Double b[][] = new Double[n][n];
        int index[] = new int[n];
        
        for (int i=0; i<n; ++i) 
            {
            	for (int j=0;j<n ;j++ ) {

            	b[i][j] = 1.0;
            	//System.out.println(b[i][i]);
            		
            	}
            }
 
 // Transform the matrix into an upper triangle
        gaussian(a, index);
 
 // Update the matrix b[i][j] with the ratios stored
        for (int i=0; i<n-1; ++i)
            for (int j=i+1; j<n; ++j)
                for (int k=0; k<n; ++k)
                   {//System.out.println(b[j][k]);
                   	b[j][k] = b[j][k] - a[j][i]*b[i][k];}
 
 // Perform backward substitutions
        for (int i=0; i<n; ++i) 
        {
            x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];
            for (int j=n-2; j>=0; --j) 
            {
                x[j][i] = b[index[j]][i];
                for (int k=j+1; k<n; ++k) 
                {
                    x[j][i] -= a[index[j]][k]*x[k][i];
                }
                x[j][i] /= a[index[j]][j];
            }
        }
        return x;
}//end of invert
   

public void calc_w()
{
	Double[][] inv_E = invert(covariance);
	Double[] mean_diff = new Double[attr];
	Double[] mean1 = new Double[attr];
	Double[] mean2 = new Double[attr];
	
	for(int j=0;j<attr;j++)
	{
		mean1[j] = classMeans[0][j];
		mean2[j] = classMeans[1][j];
		mean_diff[j] = classMeans[0][j] - classMeans[1][j];
	}
	
	w = matrix_mult(inv_E,mean_diff,4,4);

	Double[] m1,m2;
	m1 = matrix_mult(mean1,inv_E,4,4);
	m2 = matrix_mult(mean2,inv_E,4,4);
	//m1 = matrix_mult(inv_E,mean1,4,4,1);
	//m2 = matrix_mult(inv_E,mean2,4,4,1);
	double sum1=0,sum2=0;
	for(int i=0;i<attr;i++)
	{
		sum1 = sum1 + m1[i]*mean1[i];
		sum2 = sum2 + m2[i]*mean2[i];
	}


	w0 = (double)(((-1)*sum1)/2) +(sum2/2)+Math.log((double)classSum[0]/classSum[1]);

}

	public double sigmoid(Double[] xn,Double[] w1){//calculating sigmoid function

		//w1T*Xn
		double dotProduct = 0;
		for(int i=0;i<attr;i++){
			dotProduct = dotProduct + w1[i]*xn[i];
		}

		dotProduct = dotProduct + w0;

		//exponent^(-w1T*Xn)
		double ex = Math.exp(-1*dotProduct);
		ex = ex +1;//ex^() + 1

		//inverse of exp
		double inv = Math.pow(ex,-1);

		return inv;

	}

	public void test_proc(){

		for(int i=0;i<test_size;i++)
		{

			Double[] xn = new Double[attr];
			Vector<Double> x = new Vector<Double>();
			x = (Vector)test_data.elementAt(i);
			x.toArray(xn);	//convert vector to array

			double res = sigmoid(xn,w);
			double ans;
			if(res>0.5)
				ans=1.0;
			else
				ans=0.0;

			pred_label.addElement(ans);

		}
		
	}


	public void ReadTestFileBuffReader(){
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(
					"/home/tarunee/4-2/ML2/ML2 - Assignment 1/test.txt"));
			String line = reader.readLine();

			while (line != null) {
				
				String[] tokens=line.split(",");
				test_size++;
				
				//temp contains (VEctor) all the 5 attributes
				//the first 4 attributes are real numbers.
				//1.variance 2.skewness 3.curtosis 4.entropy 5.class(0/1)
				
				int i=0;
				double num =0;
				Vector<Double> attributes = new Vector<Double>();	//array of attr values
				for(String key : tokens){
					num = (Double)Double.parseDouble(key);

					if(i==attr)//for the last class labelled value
					{
						break;

					}
					attributes.addElement(num);
					i++;		
				}
				if(i==4){//---->4 is same as attr

					test_label.addElement(num);
					
				}

				test_data.addElement(attributes);	//inserting each data line.
				
				// read next line
				line = reader.readLine();
			}

			
			//System.out.println(items_info);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {

				//logReg(int numattr,double[] param,double eta,double epsilon)
		probGen pg= new probGen(4,2);
		pg.ReadFileBuffReader();
		pg.classCompute();
		pg.calc_w();


		for(int i=0;i<4;i++)
			System.out.print(pg.w[i]+"\t");
		System.out.println();
		System.out.println("w0 = "+pg.w0);


		
		pg.ReadTestFileBuffReader();
		pg.test_proc();

		int t1=0,t2=0;
		for(int i=0;i<pg.test_size;i++)
		{
			double x = (Double)pg.test_label.elementAt(i);
			if((int)x == 0)
				t1++;
			else
				t2++;
		}

		

		int accuracy=0;
		for(int i=0;i<pg.test_size;i++)
		{
			double x = (Double)pg.pred_label.elementAt(i);
			double y = (Double)pg.test_label.elementAt(i);
			if(x == y)
				accuracy++;
		}
		double acc = (double)accuracy/pg.test_size;
		///System.out.println("accuracy = "+accuracy+"total_size= "+pg.test_size); 
		System.out.println("accuracy = "+acc);


		
	}
}//end of class