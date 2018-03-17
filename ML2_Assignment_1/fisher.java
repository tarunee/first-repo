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

public class fisher{

	public Vector data;	//input points are stored here
	public int inputSize;
	public Double[] w;	//array containing parameter values --->packages for dot product can be used later.
	public int attr;	//number of attributes in input ----> specify in constructor(done in this code)or check while file reading
	public Vector<Double> target;	//saves target values
	public Double[] mean1;
	public Double[] mean2;
	public Double[][] variance;
	public int[] classSum;
	public Double  y0;

	//test variables
	public Vector test_data;	//input points are stored here
	public Vector<Double> test_label;	//saves target test label values
	public Vector<Double> pred_label;	//computed test label values
	public int test_size;

	public fisher(int a)
	{
		attr = a;

	//constructor for setting number of attribute values
	//intialisation of parameter array
	
		data = new Vector();
		inputSize = 0;
				
		w = new Double[attr];
		
		target = new Vector<Double>();
		
		mean1 = new Double[attr];
		mean2 = new Double[attr];
		variance = new Double[attr][attr];

		for (int i=0;i<attr;i++) {
			mean2[i] = 0.0;
			mean1[i] = 0.0;
			for (int j=0;j<attr;j++) {
				variance[i][j] = 0.0;
			}
			
		}

		classSum = new int[2];

		//intialize test variables here
		test_data = new Vector();
		test_label = new Vector<Double>();
		pred_label = new Vector<Double>();
		test_size = 0;

	}

	////////*****inverse******///////
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
///////*******end of invert******//////
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

	public void calc_w()
	{
		Double[][] inv_E = invert(variance);
		Double[] m = new Double[attr];
		for(int i =0 ; i<attr; i++)
		{
			m[i] = mean2[i] - mean1[i];
		}

		w = matrix_mult(inv_E,m,4,4);


	}

	
public double entropy(double[][] mat,int index){
    //binary class specific!!!
    int c1_pos=0,c1_neg=0;
    int c2_pos=0,c2_neg=0;
    for(int i=0;i<=index;i++){
        if(mat[i][1]==0) c1_pos++;
        else c1_neg++;
    }
    
    
    for(int i=index+1;i<nob;i++){
        if(mat[i][1]==0) c2_pos++;
        else c2_neg++;
    }
    double sum =0.0;
    double p_pos1 = (double)c1_pos/(double)(c1_pos+c1_neg);
    sum = sum+ p_pos1*Math.log(p_pos1);
    double p_pos2 = (double)c2_pos/(double)(c2_pos+c2_neg);
    sum = sum+ p_pos2*Math.log(p_pos2);
    double p_neg1 = (double)c1_neg/(double)(c1_pos+c1_neg);
    sum = sum+ p_neg1*Math.log(p_neg1);
    double p_neg2 = (double)c2_neg/(double)(c2_pos+c2_neg);
    sum = sum+ p_neg2*Math.log(p_neg2);
    
    return sum;
}//end of entropy

public double thresh(double[] weights){
    
    double[][] transformed_val = new double[nob][2];
    double[] entropies = new double[nob-1];
    for(int i=0;i<nob;i++){
        transformed_val[i][0] = dot_prod(weights,DM[i]);
        transformed_val[i][1] = T1[i];
        
    }
    
    //sorting the array
    for(int i=0;i<nob;i++){
        for(int j=i+1;j<nob;j++){
            if(transformed_val[i][0]>transformed_val[j][0]){
                double t1= transformed_val[i][0];
                double t2 = transformed_val[i][1];
                transformed_val[i][0] = transformed_val[j][0];
                transformed_val[i][1] = transformed_val[j][1];
                transformed_val[j][0] = t1;
                transformed_val[j][1] = t2;
                
            }//end of if
        }//end of for
    }//end of sorting
    double t ;
    double min =9999;
    double final_t=-1;
    for(int i=0;i<nob-1;i++){
        t = (transformed_val[i][0]+transformed_val[i+1][0])/2;
        entropies[i] = entropy(transformed_val,i);
        if(entropies[i]<min) 
        {
            min= entropies[i];
            final_t = t;
         
        }
        
    }//end of for
    
    return final_t;
   
}//end of thresh

	public void classCompute()
	{
		///class means
		for(int j=0;j<inputSize;j++)
		{
			Double[] xn = new Double[attr];
			Vector<Double> x = new Vector<Double>();
			x = (Vector)data.elementAt(j);
			x.toArray(xn);	//convert vector to array
			double tn = target.elementAt(j);

			for(int i=0;i<attr;i++)
			{
				mean1[i] = mean1[i] + xn[i]*(1-tn);
				mean2[i] = mean2[i] + xn[i]*tn;

			}
		}//loop for reading data

		//final class means
		for(int i=0;i<attr;i++)
		{
			mean1[i] = (double)mean1[i]/classSum[0];
			mean2[i] = (double)mean2[i]/classSum[1];

		}

		//compute class variance
		for(int i=0;i<2;i++)
		{
			Double[] new_x ;
			Double[][] varS ;
			

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
						new_x[k] = xn[k] - (mean2[k]*i);
						new_x[k] = xn[k] - (mean1[k]*(1-i));

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
							variance[k][l] = variance[k][l] + varS[k][l];
						}

					}

				}//end if that x belongs to the given class

			}//end j for each data input

		}//end i for each clas


	
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

	

	public void test_proc(){

		for(int i=0;i<test_size;i++)
		{

			Double[] xn = new Double[attr];
			Vector<Double> x = new Vector<Double>();
			x = (Vector)test_data.elementAt(i);
			x.toArray(xn);	//convert vector to array
			double sum = 0,ans;

			for(int j =0;j<attr;j++)
			{
				sum = w[j]*xn[j];
			}

			if(sum>0.5)
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
	}//end of test

	public static void main(String[] args) {

				//logReg(int numattr,double[] param,double eta,double epsilon)
		fisher fs= new fisher(4);
		fs.ReadFileBuffReader();
		fs.classCompute();
		fs.calc_w();


		for(int i=0;i<4;i++)
			System.out.print(fs.w[i]+"\t");
		System.out.println();
		
		fs.ReadTestFileBuffReader();
		fs.test_proc();

		
		int accuracy=0;
		for(int i=0;i<fs.test_size;i++)
		{
			double x = (Double)fs.pred_label.elementAt(i);
			double y = (Double)fs.test_label.elementAt(i);
			if(x == y)
				accuracy++;
		}
		double acc = (double)accuracy/fs.test_size;
		///System.out.println("accuracy = "+accuracy+"total_size= "+fs.test_size); 
		System.out.println("accuracy = "+acc);


		
	}


}