---------------------------------------------------Supervised Learning-----------------------------------------------------------
1. Linear Regression
	Hypothesis Function: 
		hθ​(x) = w0​+w1​x1​+w2​x2​+⋯+wd​xd​

	Loss Function: 
		J(θ) = MSE = (1/N) * Σ (hθ(x) - y)^2

	Optimization Algorithm: 
		BGD, SGD, Mini-BGD, Normal Equation
	



2. Logistic Regression
	Hypothesis Function: 
		hθ​(x) = P(y=1∣x;θ) = σ(w0​+w1​x1​+w2​x2​+⋯+wd​xd​) 

	Likelihood Function for 1 data set: 
		L(θ) = hθ(x)^y * (1 - hθ(x))^(1 - y)	

	Likelihood Function for all data set: 
		L(θ) = ∏[ hθ(x^(i))^y^(i) * (1 - hθ(x^(i)))^(1 - y^(i)) ]

	Loss Funtion: 
		J(θ) = -log L(θ) = - Σ [ y^(i)*log(hθ(x^(i))) + (1 - y^(i))*log(1 - hθ(x^(i))) ]

	Optimization Algorithm: 
		BGD, SGD, Mini-BGD



3. Perceptron Classifier
	Hypothesis Function
		hθ​(x) = sign(w0​+w1​x1​+w2​x2​+⋯+wd​xd​)

	Loss Function:
		J(θ) = Σ max(0, - y^(i) * (θᵀx^(i)))

	Optimization Algorithm: 
		Perceptron Learning Rule: 
			if misclassified 
				θ := θ + y * x 
			else 
				no update




4. Centroid Classifier
	Hypothesis Function
		h(x) = the class whose centroid is closest to x

	Loss Function:
		N/A

	Optimization Algorithm: 
		N/A





5. Naive Bayes Classifier
	Hypothesis Function:
       		h(x) = argmax_y  P(y) × Π_j P(x_j | y)

	Loss Function:
       		J(θ) = - Σ log P(y | x)   	# used only for evaluation not for optimizing
      
	Optimization  Algorithm:
       		Counting frequencies:
           		P(y) = class frequency
           		P(x_j | y) = feature conditional frequencies
      		 	Prediction = choose y maximizing P(y) × Π P(x_j | y)
       		




6. Support Vector Machine
	Hypothesis Function: 
		hθ​(x) = sign(w0​+w1​x1​+w2​x2​+⋯+wd​xd​)

	Loss Function:
		J(θ) = (1/2) * ||w||^2  +  C * Σ_i max(0, 1 - y_i * hθ​(x_i))

	Optimization Algorithm:
		QP solvers




7. KNN
	Hypothesis Function: 
		h(x) = majority label among the k nearest neighbors of x

	Loss Function:
		N/A

	Optimization Algorithm:
		N/A




8. Decision Tree (Gini)
	Hypothesis Function: 
		h(x) = class label stored in the leaf reached by testing x along the tree

	Loss Function:
		J(split) = Σ( (n_child / n_parent) * G(child) )

	Optimization Algorithm:
		Greedy Tree Building


---------------------------------------------------Unsupervised Learning-----------------------------------------------------------




