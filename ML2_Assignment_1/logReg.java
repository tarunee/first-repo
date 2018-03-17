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

public class logReg{
//major assumption basis function as identity funcion
//can be optimised by introducing it.
//gradient descent can be modified into stochastic also.
	//can array be returned??donno ...lite ...change if not compiled.

	//input values in vector X=<xn>
	public Vector data;	//input points are stored here
	public int inputSize;
	public Double[] w;	//array containing parameter values --->packages for dot product can be used later.
	public int attr;	//number of attributes in input ----> specify in constructor(done in this code)or check while file reading
	public Vector<Double> target;	//saves target values
	public double eta; //learning rate
	public double epsilon;	//condition for termination
	public int cls;///not used here since binary classification

	//test variables
	public Vector test_data;	//input points are stored here
	public Vector<Double> test_label;	//saves target test label values
	public Vector<Double> pred_label;	//computed test label values
	public int test_size;


	public logReg(int numattr,Double[] param,double e,double ep)
	{	
	//constructor for setting number of attribute values
	//intialisation of parameter array
	
		data = new Vector();
		inputSize = 0;
		attr = numattr;	//setting number of attributes
		
		w = new Double[attr];
		for(int i=0;i<attr;i++){	//copying initialisations
			w[i]=param[i];
		}
		target = new Vector<Double>();
		eta = e;
		epsilon = ep;

		//intialize test variables here
		test_data = new Vector();
		test_label = new Vector<Double>();
		pred_label = new Vector<Double>();
		test_size = 0;


	}

	public double sigmoid(Double[] xn,Double[] w1){//calculating sigmoid function

		//w1T*Xn
		double dotProduct = 0;
		for(int i=0;i<attr;i++){
			dotProduct = dotProduct + w1[i]*xn[i];
		}

		//exponent^(-w1T*Xn)
		double ex = Math.exp(-1*dotProduct);
		ex = ex +1;//ex^() + 1

		//inverse of exp
		double inv = Math.pow(ex,-1);

		return inv;

	}

	public Double[] gradient(Double[] w1){	//gradient function

		Double[] error= new Double[attr];	//array of size 4

		for(int i=0;i<attr;i++)
			error[i] = 0.0;

		//i is index for all data-inputs
		//j is index for error[j]
		//calculate error[i] for all data inputs.

		for(int i=0;i<attr;i++)	//outer loop
		{
			for(int j=0;j<inputSize;j++)	//inner loop
			{
				Double[] xn = new Double[attr];
				Vector<Double> x = new Vector<Double>();
				x = (Vector)data.elementAt(j);
				x.toArray(xn);	//convert vector to array
				double yn = target.elementAt(j);
				error[i] = error[i] + ((sigmoid(xn,w1)-yn)*xn[i]);
			}		
		}

		return error;

	}

	public double norm(Double[] w1,Double[] w){	//assuming euclidean norm <-- can be changed.

		double sum =0;
		for(int i=0;i<attr;i++){
			sum = sum + ((w[i]-w1[i])*(w[i]-w1[i]));
		}

		sum = Math.sqrt(sum);
		return sum;
	}

	public void gradientDescent(){//Gradient Descendent Function
	//iteratively calculate w function

		Double[] w1 = new Double[attr];	//array for saving new parameter valus
		for(int i=0;i<attr;i++){		//intialising new array to zero
			w1[i] = 0.0;
		}


		Double[] w2 = new Double[attr];	//array for saving new parameter valus
		for(int i=0;i<attr;i++){		//intialising new array to zero
			w2[i] = 0.0;
		}

		int iteration=1;
		while(norm(w1,w)>epsilon){	//condition for termination

			Double[] g = gradient(w1);	
			for(int i=0;i<attr;i++){

				w1[i] = w[i] - eta*(g[i]);

			}//end of for

			for(int i=0;i<attr;i++){		//changing to new array
			w2[i] = w1[i];
			}

			iteration++;

			//w1=w and w=w2
			for(int i=0;i<attr;i++){		//changing to new array
			w1[i] = w[i];
			}
			for(int i=0;i<attr;i++){		//changing to new array
			w[i] = w2[i];
			}



		}//end of while
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

	
	//reading test data

	public void test_proc(){

		for(int i=0;i<test_size;i++)
		{

			Double[] xn = new Double[attr];
			Vector<Double> x = new Vector<Double>();
			x = (Vector)test_data.elementAt(i);
			x.toArray(xn);	//convert vector to array
			double val = sigmoid(xn,w);
			//System.out.println("val = "+val);
			if(val>=0.5)
				pred_label.addElement(1.0);
			else
				pred_label.addElement(0.0);
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

		Double[] w0 = new Double[4];

		for(int i=0;i<4;i++)
			w0[i] =0.01;

		//logReg(int numattr,double[] param,double eta,double epsilon)
		logReg lg= new logReg(4,w0,0.0001,0.005);
		lg.ReadFileBuffReader();

		
		lg.gradientDescent();

		for(int i=0;i<4;i++)
			System.out.print(lg.w[i]+"\t");
		System.out.println();


		
		lg.ReadTestFileBuffReader();
		lg.test_proc();


		int accuracy=0;
		for(int i=0;i<lg.test_size;i++)
		{
			double x = (Double)lg.pred_label.elementAt(i);
			double y = (Double)lg.test_label.elementAt(i);
			if(x == y)
				accuracy++;
		}
		double acc = (double)accuracy/lg.test_size;
		///System.out.println("accuracy = "+accuracy+"total_size= "+lg.test_size); 
		System.out.println("accuracy = "+acc);

		
	}
}
