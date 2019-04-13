    def etapa3(regioes = 5, vagalumes = 80, geracoes = 100):
    
        im1 = ffly.lplHisteq(self.imageGray)
        
        H = ffly.psrGrayHistogram(im1)
        
        intensidades = ffly.lplFirefly(vagalumes, regioes, 1, 0.97, 1, geracoes, H)
        
        #imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        altura, largura = self.imageGray.shape
        
        imagemNova = np.copy(self.imageGray)
        
        for y in range(altura):
            for x in range(largura):
                valor = self.comparaIntensidade(im1[y, x], intensidades)
                imagemNova[y, x] = valor
                imagemNova[y, x] = valor
                imagemNova[y, x] = valor
        
        cv2.imwrite("img.png", imagemNova)
        
        intensidades.append(0)
        
        intensidades.sort()
        
        print(intensidades)
        
        return intensidades

    def comparaIntensidade(valor, vetor):
        ret = valor
        for i in range(0, len(vetor)):
            if i == 0:
                if valor < vetor[i]:
                    ret = 0
            else:
                if valor < vetor[i] and valor >= vetor[i - 1]:
                    ret = vetor[i - 1]
                elif valor >= vetor[i]:
                    ret = vetor[i]
        return ret