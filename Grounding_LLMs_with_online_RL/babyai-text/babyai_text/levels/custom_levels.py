from babyai.levels.levelgen import *

class Level_GoToPickupOnly(LevelGen):
    """
    Environment with only GoTo and PickUp tasks in a single room setup.
    Tasks are randomly chosen between going to an object or picking up an object.
    """
    
    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We only select from 'goto' and 'pickup'
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto', 'pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):
        action = self._rand_elem(self.action_kinds)
        mission_accepted = False
        all_objects_reachable = False
        
        if action == 'goto':
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
                
                mission_accepted = True
                # Uncomment and modify if you need to exclude certain combinations
                # mission_accepted = not (self.exclude_substrings())
                
        elif action == 'pickup':
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(num_distractors=self.num_dists + 1, all_unique=False)
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                while str(obj.type) == 'door':
                    obj = self._rand_elem(objs)
                self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))
                
                mission_accepted = True
                # Uncomment and modify if you need to exclude certain combinations
                # mission_accepted = not (self.exclude_substrings())

    # Optional: Implement if you need to exclude specific object-color combinations
    # def exclude_substrings(self):
    #     # True if contains excluded substring
    #     list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

    #     for sub_str in list_exclude_combinaison:
    #         if sub_str in self.instrs.surface(self):
    #             return True
    #     return False

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

# Register the level
register_levels(__name__, globals())