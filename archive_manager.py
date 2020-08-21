import json
from os import makedirs
from os.path import exists
import numpy as np

from properties import RESULTS_PATH, ARCHIVE_THRESHOLD
from utils import get_distance

class Archive:

    def __init__(self):
        self.archive = list()
        self.archived_seeds = set()
        self.threshold = None

    def get_archive(self):
        return self.archive

    def update_archive(self, ind):
        if ind not in self.archive:
            if len(self.archive) == 0:
                self.archive.append(ind)
                self.archived_seeds.add(ind.seed)
            else:
                # Find the member of the archive that is closest to the candidate.
                closest_archived = None
                d_min = np.inf
                i = 0
                while i < len(self.archive):
                    distance_archived = get_distance(ind.member.purified, self.archive[i].member.purified)
                    if distance_archived < d_min:
                        closest_archived = self.archive[i]
                        d_min = distance_archived
                    i += 1
                # Decide whether to add the candidate to the archive
                # Verify whether the candidate is close to the existing member of the archive
                # Note: 'close' is defined according to a user-defined threshold
                if d_min > ARCHIVE_THRESHOLD:
                    # Add the candidate to the archive if it is distant from all the other archive members
                    self.archive.append(ind)
                    self.archived_seeds.add(ind.seed)


    def get_seeds(self):
        seeds = set()
        for ind in self.get_archive():
            seeds.add(ind.seed)
        return seeds

    def get_fitnesses(self):
        fitnesses = list()
        stats = [None]*4
        for ind in self.get_archive():
            fitnesses.append(ind.ff)

        stats[0] = np.min(fitnesses)
        stats[1] = np.max(fitnesses)
        stats[2] = np.mean(fitnesses)
        stats[3] = np.std(fitnesses)
        return stats


    def create_report(self, generation):
        # Retrieve the solutions belonging to the archive.
        solution = [ind for ind in self.archive]
        N = (len(solution))


        # Obtain misclassified member of an individual on the frontier.
        misclassified_inputs = []
        # Obtain correctly classified member of an individual on the frontier.
        for ind in solution:
            if (ind.member.predicted_label != ind.member.expected_label):
                misclassified_inputs.append(ind)

        print("Final solution N is: " + str(N))

        seeds = self.get_seeds()
        stats = self.get_fitnesses()

        report = {
            'archive_len': str(N),
            'misclassified': str(len(misclassified_inputs)),
            'covered seeds': str(len(seeds)),
            'min_fitness': str(stats[0]),
            'max_fitness': str(stats[1]),
            'avg_fitness': str(stats[2]),
            'std_fitness': str(stats[3]),
            }

        if not exists(RESULTS_PATH):
            makedirs(RESULTS_PATH)
        dst = RESULTS_PATH + '/report_'+str(generation)+'.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()

