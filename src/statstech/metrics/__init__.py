class Metrics:
    "Class of metrics utilised for evaluating regression and classificators models"


    """"""
    def RSS(self, x: list[list[float]], y: list[float], f) -> float:
        errors = []
        n = len(x)
        for i in range(n):
            y_pred = f(x[i])
            y_true = y[i]
            sqrd_error = (y_pred - y_true)**2
            errors.append(sqrd_error)
        total_error = sum(errors)
        return total_error
    
    """a"""
    def MSE(self, x: list[list[float]], y: list[float], f) -> float:
        total_error = self.RSS(x, y, f)
        n = len(x)
        return total_error/n
        
    """"""
    def MAE(self, x: list[list[float]], y: list[float], f) -> float:
        errors = []
        n = len(x)
        for i in range(n):
            abs_error = abs(f(x[i]) - y[i])
            errors.append(abs_error)
        total = sum(errors)
        return total/n

    """"""    
    def RMSE(self, x: list[list[float]], y: list[float], f) -> float:
        mse = self.MSE(x, y, f)
        return mse**0.5

    def R2_score(self, x: list[list[float]], y: list[float], f) -> float:
        rss = self.RSS(x, y, f)
        n = len(x)
        media = sum(y)/n 
        errors = []
        for i in range(n):
            y_pred = media
            y_true = y[i]
            sqrd_error = (y_true - y_pred)**2
            errors.append(sqrd_error)
        tss = sum(errors)
        if tss != 0:
            return 1-rss/tss
        else:
            raise ValueError("R² is undefined when TSS = 0")