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
		BGD, SGD, Mini-BGD, Newton Method



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