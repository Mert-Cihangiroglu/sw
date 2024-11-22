# client_class_assignments.py

class ClientClassAssignments:
    """A collection of predefined class assignments for federated clients in different configurations."""

    @staticmethod
    def no_overlap():
        """Each client gets a unique single class with no overlap."""
        return {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5],
            6: [6],
            7: [7],
            8: [8],
            9: [9]
        }

    @staticmethod
    def two_class_overlap():
        """Each client gets two classes with adjacent clients sharing one class."""
        return {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
            4: [4, 5],
            5: [5, 6],
            6: [6, 7],
            7: [7, 8],
            8: [8, 9],
            9: [9, 0]
        }

    @staticmethod
    def full_overlap():
        """Each client has access to all classes (IID setup)."""
        all_classes = list(range(10))
        return {client_id: all_classes for client_id in range(10)}

    @staticmethod
    def clustered_assignment():
        """Clients divided into clusters with shared common classes."""
        return {
            0: [0, 1, 2],
            1: [0, 1, 3],
            2: [0, 1, 4],
            3: [0, 1, 5],
            4: [0, 1, 6],
            5: [7, 8, 9],
            6: [7, 8, 0],
            7: [7, 8, 1],
            8: [7, 8, 2],
            9: [7, 8, 3]
        }

    @staticmethod
    def increasing_variety():
        """Each client has an increasing number of classes, creating a gradient of class diversity."""
        return {
            0: [0],
            1: [0, 1],
            2: [0, 1, 2],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 3, 4],
            5: [5, 6],
            6: [5, 6, 7],
            7: [5, 6, 7, 8],
            8: [5, 6, 7, 8, 9],
            9: list(range(10))
        }

    @staticmethod
    def cross_client_coverage():
        """Each client receives two classes, ensuring each class is covered by exactly two clients."""
        return {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
            4: [4, 5],
            5: [5, 6],
            6: [6, 7],
            7: [7, 8],
            8: [8, 9],
            9: [9, 0]
        }

    @staticmethod
    def few_comprehensive_clients():
        """Some clients receive all classes, while others specialize in specific classes."""
        return {
            0: list(range(10)),
            1: [0, 1],
            2: [2, 3],
            3: [4, 5],
            4: [6, 7],
            5: list(range(10)),
            6: [8, 9],
            7: [1, 2],
            8: [3, 4],
            9: [5, 6]
        }

# Example usage:
# To use one of these settings in your code, simply import the class and call the desired method:
# from client_class_assignments import ClientClassAssignments
# client_assignments = ClientClassAssignments.no_overlap()