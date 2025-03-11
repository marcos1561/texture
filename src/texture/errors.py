class AllMatricesNullError(Exception):
    def __init__(self):
        super().__init__("All matrices are null!")

if __name__ == "__main__":
    raise AllMatricesNullError()