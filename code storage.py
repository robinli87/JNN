#code storage


    def SGD(self, sample_input, sample_output):
        L = len(self.W)
        perturbation = 0.001
        for l in range(0, L):
            for j in range(0, self.structure[l]):
                for k in range(0, self.structure[l+1]):
                    W1 = self.copy_weights()
                    W1[l][j][k] += perturbation
                    upper = self.loss(sample_input, sample_output, W1, self.J, self.B)
                    W2 = self.copy_weights()
                    W2[l][j][k] -= perturbation
                    lower = self.loss(sample_input, sample_output, W2, self.J, self.B)

                    gradient = (upper - lower) / (2 * perturbation)

                    self.W[l][j][k] -= self.learning_rate * gradient

        for l in range(0, L):
            for k in range(0, self.structure[l+1]):
                B1 = self.copy_biases()
                B2 = self.copy_biases()

                B1[l][k] += perturbation
                B2[l][k] -= perturbation

                upper = self.loss(sample_input, sample_output, self.W, self.J, B1)
                upper = self.loss(sample_input, sample_output, self.W, self.J, B2)
                print(gradient)
                gradient = (upper - lower) / (2 * perturbation)

                self.B[l][k] -= self.learning_rate * gradient


    def copy_weights(self):
        new_W = []
        for l in range(0, len(self.structure)-1):
            new_W.append(np.array(self.W[l], copy=True))
        # for l in range(0, len(self.structure)-1):
        #     for j in range(0, self.structure[l]):
        #         for k in range(0, self.structure[l+1]):
        #             new_W[l][j][k] = Decimal(self.W[l][j][k])

        return(new_W)

    def copy_biases(self):
        new_B = []
        for l in range(0, len(self.structure)-1):
            new_B.append(np.zeros((self.structure[l+1])))

        for l in range(0, len(self.structure)-1):
            for k in range(0, self.structure[l+1]):
                new_B[l][k] = Decimal(self.B[l][k])

        return(new_B)

    def copy_jumpers(self):
        new_J = []
        for l in range(2, len(self.structure)-1):
            new_J.append([])
            for m in range(0, l-1):
                new_J[l-2].append(np.zeros((self.structure[l], self.structure[l+1])))

        for l in range(0, len(self.structure)-1):
            for m in range(0, l-1):
                for j in range(0, self.structure[l]):
                    for k in range(0, self.structure[l+1]):
                        new_J[l][m][j][k] = self.J[l][m][j][k]

        return(new_J)
