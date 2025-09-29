from jax import jit, vmap, jacfwd, grad
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import timeit

jax.config.update('jax_enable_x64', True)

class MechanismBuilder:

    def __init__(self, instance_name:str, world_pose=(0, 0, 0), gravity=(0, -9.81)):
        self.mechanism_name = instance_name
        self.world = {"pose": world_pose, "points": {"O": (0.0, 0.0)}, "frames": {}}
        self.bodies = []
        self.joints = []
        self.drivers = []
        self.options = {"gravity": gravity}        

    def add_world_point(self, name:str, ground_point:list[float]|tuple[float, float]) -> None:
        x, y = ground_point
        self.world["points"][name] = tuple((float(x), float(y)))

    def add_body(self, name:str, points:dict, initial_configuration:list, mass:float, inertia_zz:float, dynamic:bool = True) -> None:
        #TODO add checks for sanitizing inputs
        
        self.bodies.append(
            {
                "name": name,
                "points": points,
                "initial_configuration": initial_configuration,
                "mass": mass,
                "inertia_zz": inertia_zz,
                "dynamic": dynamic
            }
        )

        print(f"The body has been added and can be viewed with expose_dict(\"bodies\")")

    def add_joint(self, name:str, joint_type:str, endpoint_1:dict, endpoint_2: dict, constraint_arguments:dict) -> None:
        #TODO add checks for sanitizing inputs
        joint_type_clean = joint_type.lower().strip()
        joint_types = ("revolute", "distance", "translation", "prismatic",
                                "angle", "gear", "composite", "weld")
        joint_1_dof = ("revolute", "translation", "gear", "angle")
        joint_2_dof = ("distance", "prismatic")
        
        if joint_type_clean in joint_types:
            if joint_type_clean in joint_1_dof:
                rows = 2
            elif joint_type_clean in joint_2_dof:
                rows = 1
            elif joint_type_clean == "weld":
                rows = 0
            else:
                rows = -1

            self.joints.append(
                {
                    "name": name,
                    "joint_type": joint_type.lower().strip(),
                    "endpoint_1": endpoint_1,
                    "endpoint_2": endpoint_2,
                    "constraint_arguments": constraint_arguments,
                    "rows": rows
                }
            )

    def set_gravity(self, gravity: tuple[float, float]) -> None:
        x, y = gravity
        self.options["gravity"] = tuple((float(x), float(y)))

    def compile(self):
        float_dtype = jnp.float64
        int_dtype = jnp.int32

        dyn_bodies = [body for body in self.bodies if body.get("dynamic", True)]
        body_id_by_name = {body["name"]: index for index, body in enumerate(dyn_bodies)}
        body_by_name = {body["name"]: body for body in self.bodies}
        
        mass = jnp.array([float(body["mass"]) for body in dyn_bodies], dtype=float_dtype)
        inertia_zz = jnp.array([float(body["inertia_zz"]) for body in dyn_bodies], dtype=float_dtype)
        gravity_x, gravity_y = self.options.get("gravity")
        gravity = jnp.array((float(gravity_x), float(gravity_y)), dtype=float_dtype)


        number_of_joints = len(self.joints)
        number_of_bodies = len(dyn_bodies)

        joint_map = {
            "revolute": 0,
            "distance": 1,
            "translation": 2,
            "prismatic": 3,
            "angle": 4,
            "gear": 5,
            "composite": 6,
            "weld": 7
            }

        joint_type_id_list = []
        joint_i_list = []
        joint_j_list = []
        si_list = []
        sj_list = []
        distance_list = []

        def _resolve_endpoint(endpoint: dict) -> tuple[int, tuple[float, float]]:
            body_name = endpoint["body"]
            point_name = endpoint["point"]

            if body_name == "world":
                endpoint_x, endpoint_y = self.world["points"][point_name]
                return -1, (float(endpoint_x), float(endpoint_y))

            body_id = body_id_by_name[body_name]
            endpoint_x, endpoint_y = body_by_name[body_name]["points"][point_name]
            return body_id, (float(endpoint_x) ,float(endpoint_y))
        
        for joint in self.joints:
            joint_string = joint["joint_type"].lower().strip()
            joint_type = joint_map[joint_string]

            joint_i, si = _resolve_endpoint(joint["endpoint_1"])
            joint_j, sj = _resolve_endpoint(joint["endpoint_2"])

            joint_type_id_list.append(joint_type)
            joint_i_list.append(joint_i)
            joint_j_list.append(joint_j)
            si_list.append(si)
            sj_list.append(sj)

            if joint_type == joint_map["distance"]:
                distance_value = float(joint.get("constraint_arguments", {}).get("distance", 0.0))
                distance_list.append(distance_value)
            else:
                distance_list.append(0.0)

        if number_of_joints == 0:
            joint_type_id = jnp.zeros((0,), dtype=int_dtype)
            joint_i = jnp.zeros((0,), dtype=int_dtype)
            joint_j = jnp.zeros((0,), dtype=int_dtype)
            si_xy = jnp.zeros((0, 2), dtype=float_dtype)
            sj_xy = jnp.zeros((0, 2), dtype=float_dtype)
            distance = jnp.zeros((0,), dtype=float_dtype)
            print(f"Arrays of size zero were created since number of joints = {number_of_joints}")
        else:
            joint_type_id = jnp.array(joint_type_id_list, dtype=int_dtype)
            joint_i = jnp.array(joint_i_list, dtype=int_dtype)
            joint_j = jnp.array(joint_j_list, dtype=int_dtype)
            si_xy = jnp.array(si_list, dtype=float_dtype).reshape(number_of_joints, 2)
            sj_xy = jnp.array(sj_list, dtype=float_dtype).reshape(number_of_joints, 2)
            distance = jnp.array(distance_list, dtype=float_dtype)

        parameters = {
            "mass": mass,
            "inertia_zz": inertia_zz,
            "gravity": gravity,
            "joint_type_id": joint_type_id,
            "joint_i": joint_i,
            "joint_j": joint_j,
            "si_xy": si_xy,
            "sj_xy": sj_xy,
            "distance": distance
        }

        number_of_coordinates = 3 * number_of_bodies

        if number_of_joints == 0:
            row_counts = jnp.zeros((0,) dtype=int_dtype)
            row_offset = jnp.zeros((0,) dtype=int_dtype)
            number_of_rows = 0

        else:
            row_counts_list = []
            for joint in joint_type_id_list:
                if joint == joint_map["revolute"]:
                    row_counts_list.append(joint_map.get("revolute"))
                elif joint == joint_map["distance"]:
                    row_counts_list.append(joint_map.get("distance"))
                else:
                    row_counts_list.append(0)
            
            row_counts = jnp.array(row_counts_list, dtype=int_dtype)
            row_offset = jnp.cumsum(row_counts) - row_counts
            number_of_rows = int(row_counts.sum())

        layout = {
            "num_bodies": number_of_bodies,
            "num_joints": number_of_joints,
            "num_coord": number_of_coordinates,
            "body_id_by_name": body_id_by_name,
            "coord_slices": coordinate_slices,
            ""
        }











        return parameters




    def __call__(self, *args, **kwds):
        pass

    def validate_mechanism_setup(self, type:str = "all"):
        pass
        """ if type == "all":
            for key, value in self.world.items():
                self.assertTrue(key is )
                if key == "pose":
                    assert is """

    def expose_dict(self, type="world"):
        if type == "world":
            return self.world
        elif type == "bodies":
            return self.bodies
        elif type == "joints":
            return self.joints
        elif type == "drivers":
            return self.drivers
        elif type == "options":
            return self.options
        else:
            print(f"Wrong input. The input was {type}")





if __name__ == "__main__":
    test = MechanismBuilder("test")
    
    point_dict = {"A": [0.0, 0.0], "B": [1.0, 0.0]}
    test.add_body("test", point_dict, [0,0,0], 3, 0.5)
    test.add_joint("J1", "revolute",
                {"body":"test","point":"A"},
                {"body":"world","point":"O"},
                {})
    test.add_joint("J2", "distance",
                {"body":"test","point":"B"},
                {"body":"world","point":"O"},
                {"distance": 5})


    params = test.compile()



    """ point_dict = {"A": [0.0, 0.0],
                  "B": [1.0, 0.0]}
    test.add_body("test", point_dict, [0,0,0], 3, 0.5)
    
    endpoint_1 = {"body": "test", "point": "A"}
    endpoint_2 = {"body": "world", "point": "O"}
    test.add_joint("joint_1_between body1 and ground", "revolute", endpoint_1, endpoint_2, {"distance": 2})
    parameters = test.compile()
    testing = 0
    #def add_joint(self, name:str, joint_type:str, endpoint_1:dict, endpoint_2: dict, constraint_arguments:dict)
    #print(test.expose_dict("bodies")) """

    pass

    """ 
    system = {}


    system['body_1'] = {} """