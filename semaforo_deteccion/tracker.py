import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.nextObjectID = 0
        self.objects = {}    # id -> centroid (cX, cY)
        self.bboxes = {}     # id -> (x1,y1,x2,y2)
        self.disappeared = {}# id -> frames desaparecido
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]

    @staticmethod
    def _centroid_from_bbox(b):
        x1, y1, x2, y2 = b
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

    def update(self, rects):
        # rects: lista de (x1,y1,x2,y2)
        if len(rects) == 0:
            # Incrementa "desaparecido" y elimina si excede el umbral
            to_remove = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    to_remove.append(objectID)
            for objectID in to_remove:
                self.deregister(objectID)
            return self.bboxes.copy()

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, b in enumerate(rects):
            inputCentroids[i] = self._centroid_from_bbox(b)

        # Si no hay objetos previos, registra todos
        if len(self.objects) == 0:
            for i, b in enumerate(rects):
                self.register(inputCentroids[i], b)
            return self.bboxes.copy()

        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()))

        # Matriz de distancias Euclidianas (sin SciPy)
        # D.shape = (num_objs_existentes, num_rects_nuevos)
        diff = objectCentroids[:, None, :] - inputCentroids[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = tuple(inputCentroids[col])
            self.bboxes[objectID] = rects[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # No asignados
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, len(rects))).difference(usedCols)

        # Más objetos previos que nuevos -> incrementa desaparecidos
        if D.shape[0] >= len(rects):
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
        else:
            # Más detecciones nuevas -> registra nuevos IDs
            for col in unusedCols:
                self.register(tuple(inputCentroids[col]), rects[col])

        return self.bboxes.copy()
