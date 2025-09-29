
        def _dynamic_bodies() -> list[dict]:
            bodies = [body for body in self.bodies if body.get("dynamic", True)]
            return bodies

        def _A_bodies_arrays(dynamic_bodies):
            mass_array = [float(body["mass"]) for body in dynamic_bodies]
            inertia_array = [float(body["inertia_zz"]) for body in dynamic_bodies]

            return mass_array, inertia_array

        def _gravity_vector():
            gravity_x, gravity_y = self.options.get("gravity")
            return float(gravity_x), float(gravity_y)
        
        def _compile_A_bodies(float_dtype=jnp.float64) -> dict:
            dynamic_bodies = _dynamic_bodies()
            gravity_x, gravity_y = _gravity_vector()
            mass_array, inertia_array = _A_bodies_arrays(dynamic_bodies)

            A = {
                "mass": jnp.array(mass_array, dtype=float_dtype),
                "inertia_zz": jnp.array(inertia_array, dtype=float_dtype),
                "gravity": jnp.array((gravity_x, gravity_y), dtype=float_dtype)
            }
            return A

        def _joint_type_map() -> dict[str, int]:
            joint_map = {
                "revolute": 0,
                "distance": 1
                #"translation": 2,
                #"prismatic": 3,
                #"angle": 4,
                #"gear": 5,
                #"composite": 6,
                #"weld": 7,
            }
            return joint_map

        def _build_body_id_map(dyn_bodies: list[dict]) -> dict[str, int]:
            body_map_dict = {body["name"]: index for index, body in enumerate(dyn_bodies)}

            return body_map_dict

        def _resolve_endpoint(endpoint: dict, body_id_by_name: dict) -> tuple[int, tuple[float, float]]:
            
            body_name = endpoint["body"]
            point_name = endpoint["point"]

            if body_name == "world":
                end_x, end_y = self.world["points"][point_name]

                return -1, (float(end_x), float(end_y))
            
            body_id = body_id_by_name[body_name]
            end_x, end_y = body_id_by_name[body_name]["points"][point_name]
            return body_id, (float(end_x), float(end_y))

        def _A_joints_lists(dyn_bodies: list[dict]) -> tuple[list]:
            type_map = _joint_type_map()
            body_id_by_name = _build_body_id_map(dyn_bodies)

            joint_type_id_list = []
            joint_1_list = []
            joint_2_list = []
            point_1_list = []
            point_2_list = []
            distance_list = []

            for joint in self.joints:
                joint_string = joint["joint_type"].lower().strip()
                joint_type_id = type_map[joint_string]

                joint_1_id, point_1_id = _resolve_endpoint(joint["endpoint_1"], body_id_by_name)
                joint_2_id, point_2_id = _resolve_endpoint(joint["endpoint_2"], body_id_by_name)

                joint_type_id_list.append(joint_type_id)
                joint_1_list.append(joint_1_id)
                joint_2_list.append(joint_2_id)
                point_1_list.append((float(point_1_id[0]), float(point_1_id[1])))
                point_2_list.append((float(point_2_id[0]), float(point_2_id[1])))

                if joint["constraint_arguments"] == "distance":
                    distance_list.append(float(joint["constraint_arguments"]["distance"]))
                else:
                    distance_list.append(0.0)

            number_of_joints = len(self.joints)

            return (joint_type_id_list, joint_1_list, joint_2_list,
                    point_1_list, point_2_list, distance_list,
                    number_of_joints, body_id_by_name)


        def _compile_A_joints(dyn_bodies: list[dict], float_dtype=jnp.float64) -> dict[jax.typing.ArrayLike]:
            (
                joint_type_id_list,
                joint_1_list,
                joint_2_list,
                point_1_list,
                point_2_list,
                distance_list,
                number_of_joints,
                body_id_by_name
                ) = _A_joints_lists(dyn_bodies=dyn_bodies)

            if number_of_joints == 0:
                A_joints = {
                    "joint_type_id": jnp.zeros((0,),    dtype=jnp.int32),
                    "joint_1": jnp.zeros((0,),    dtype=jnp.int32),
                    "joint_2": jnp.zeros((0,),    dtype=jnp.int32),
                    "point_1": jnp.zeros((0, 2),    dtype=float_dtype),
                    "point_2": jnp.zeros((0, 2),    dtype=float_dtype),
                    "distance": jnp.zeros((0,),    dtype=float_dtype),
                }
                return A_joints, body_id_by_name
            
            A_joints = {
                "joint_type_id": jnp.array(joint_type_id_list,    dtype=jnp.int32),
                "joint_1": jnp.array(joint_1_list,    dtype=jnp.int32),
                "joint_2": jnp.array(joint_2_list,    dtype=jnp.int32),
                "point_1": jnp.array(point_1_list,    dtype=float_dtype),
                "point_2": jnp.array(point_2_list,    dtype=float_dtype),
                "distance": jnp.array(distance_list,    dtype=float_dtype),
            }
            return A_joints, body_id_by_name

        A_bodies = _compile_A_bodies(float_dtype=jnp.float64)
        A_joints = _compile_A_joints(float_dtype=jnp.float64)

        parameters = {**A_bodies, **A_joints}
        return parameters
        


        """ self.mechanism_name
        self.world
        self.bodies
        self.joints
        self.drivers
        self.options
        

        def _body_id_by_name() -> dict:


        return parameters, layout, initial_config, report """


number_of_dyn_bodies = len(dyn_bodies)