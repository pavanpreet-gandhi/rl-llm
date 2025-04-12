from babyai.levels.levelgen import *


class Level_GoTo(LevelGen):
    """
    GoTo instruction from MixedTrainLocal.
    The agent does not need to move objects around.
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: GoTo
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj = self._rand_elem(objs)
            self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_Pickup(LevelGen):
    """
    PickUp instruction from MixedTrainLocal.
    The agent does not need to move objects around.
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: PickUp
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj = self._rand_elem(objs)
            while str(obj.type) == 'door':
                obj = self._rand_elem(objs)
            self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_Open(LevelGen):
    """
    Open instruction from MixedTrainLocal.
    The agent needs to open a locked door between two rooms.
    There are 2 rooms and the door in between is locked.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: Unlock
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=2,  # Always 2 columns for Open task
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            color_door = self._rand_elem(['yellow', 'green', 'blue', 'purple'])  # red and grey excluded
            self.add_locked_room(color_door)
            self.connect_all()

            for j in range(self.num_rows):
                for i in range(self.num_cols):
                    if self.get_room(i, j) is not self.locked_room:
                        self.add_distractors(i, j, num_distractors=self.num_dists, all_unique=False)

            # The agent must be placed after all the object to respect constraints
            while True:
                self.place_agent()
                start_room = self.room_from_pos(*self.agent_pos)
                # Ensure that we are not placing the agent in the locked room
                if start_room is self.locked_room:
                    continue
                break

            all_objects_reachable = self.check_objs_reachable(raise_exc=False)

            color_in_instr = self._rand_elem([None, color_door])

            desc = ObjDesc('door', color_in_instr)
            self.instrs = OpenInstr(desc)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_PutNext(LevelGen):
    """
    PutNext instruction from MixedTrainLocal.
    The agent needs to put one object next to another.
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: PutNext
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['putnext'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_1 = self._rand_elem(objs)
            while str(obj_1.type) == 'door':
                obj_1 = self._rand_elem(objs)
            desc1 = ObjDesc(obj_1.type, obj_1.color)
            obj_2 = self._rand_elem(objs)
            if obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                obj1s, poss = desc1.find_matching_objs(self)
                if len(obj1s) < 2:
                    # if obj_1 is the only object with this description obj_2 has to be different
                    while obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                        obj_2 = self._rand_elem(objs)
            desc2 = ObjDesc(obj_2.type, obj_2.color)
            self.instrs = PutNextInstr(desc1, desc2)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_PickUpSeqGoTo(LevelGen):
    """
    Pick up then/before go to sequence instruction from MixedTrainLocal.
    The agent needs to pick up an object and then go to another object (or the reverse order).
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: PickUp, GoTo, Seq
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pick up seq go to'],
            instr_kinds=['seq1'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == 'door':
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            type_instr = self._rand_elem(['Before', 'After'])

            if type_instr == 'Before':
                self.instrs = BeforeInstr(instr_a, instr_b)
            else:
                self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_GoToTest(LevelGen):
    """
    GoTo instruction from MixedTrainLocal.
    The agent does not need to move objects around.
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: GoTo
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj = self._rand_elem(objs)
            self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # Now checks if any of the desired substrings are included
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]
        
        # Return False if any substring is found (don't exclude)
        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return False
        
        # Return True to exclude if none of the substrings are found
        return True

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


class Level_PickupTest(LevelGen):
    """
    PickUp instruction from MixedTrainLocal.
    The agent does not need to move objects around.
    There is only one room.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door

    Competencies: PickUp
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False
        
        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj = self._rand_elem(objs)
            while str(obj.type) == 'door':
                obj = self._rand_elem(objs)
            self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # Now checks if any of the desired substrings are included
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]
        
        # Return False if any substring is found (don't exclude)
        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return False
        
        # Return True to exclude if none of the substrings are found
        return True

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0


register_levels(__name__, globals())
