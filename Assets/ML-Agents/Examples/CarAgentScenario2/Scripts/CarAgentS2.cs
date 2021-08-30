/**
 * @file CarAgentS2.cs
 * @author Lucas Steffanutto
 * @date 02/07/2021
 * @brief KARlab Use Case 3: Unity Proof of Concept of an Autonomous Parking System with Reinforcement Learning
 * 
 * @copyright Copyright (c) Advans Group
 */

using UnityEditor;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

/**
 * @brief Serializable for each Wheels collider of the Agent to then apply motor, brake and steering to the front wheels: [WheelCollider Unity Tutorial](https://docs.unity3d.com/Manual/WheelColliderTutorial.html)
 * @note motor, steering and brake has to be associate only to the front wheels of the kart
 */
[System.Serializable]
public class AxleInfo2
{
    public WheelCollider leftWheel; /**<Associate a kart left wheel in the Unity Editor */
    public WheelCollider rightWheel;/**<Associate a kart right wheel in the Unity Editor */
    public bool motor;              /**< Bool to chose if Throttle/Motor Torque is applied to the associate wheels */
    public bool steering;           /**< Bool to chose if Steering Angle  is applied to the associate wheels */
    public bool brake;              /**< Bool to chose if Brake is applied to the associate wheels */
}

/**
 * @brief Modelisation of the KARlab Use Case 3 with the Unity plugin [ML-Agent release 17] (https://github.com/Unity-Technologies/ml-agents)
 *
 * This class allowed to modelise and train the kart to parking thanks to Reinforcement Learning.
 */
public class CarAgentS2 : Agent
{
    public List<AxleInfo2> axleInfos;    /**<Front and Back wheels of the kart */

    /** \name Max values of Motor Torque, Steering Angle and Brake Torque that can be apply on the front wheels */
    ///@{
    public float maxMotorTorque = 50f;
    public float maxSteeringAngle = 20f;
    public float maxBrakeTorque;
    ///@}

    /** \name OBSERVATIONS for the RL kart model training */
    ///@{
    private float velocityXNorm;                    /**<Agent normalized velocity on the X axis*/
    private float velocityZNorm;                    /**<Agent normalized velocity on the Z axis*/
    private float headingAngleNorm;                 /**<Agent normalized heading angle forward direction*/
    private Vector3 agentPositionNorm;              /**<Agent Position normalized (x,y,z) from environment center*/
    private Vector3 parkingPositionNorm;            /**<Parking Target Position normalized (x,y,z) from environment center*/
    private Vector3 distanceGoalAgentNorm;          /**<3D normalized distance Vector Position normalized (x,y,z) to the parking target*/
    ///@}

    /** \name All the boolean to show model Logs in the Unity Editor Console */
    ///@{
    public bool showEpisodes;                       /**<Show the Episodes number and rewards*/
    public bool showAgentLogs;                      /**<Show the agent position, velocity etc.*/
    public bool showHeuristicLogs;                  /**<Show the actions buffer sent to the Agent during Heuristic mode*/
    public bool showHeadingLogs;                    /**<Show all the heading variables*/
    public bool showTimeLogs;                       /**<Show the time*/
    bool showRewardLogs = false;                    /**<Show the Agent cumulative rewards evolution during time*/
    public bool showLogsPythonAPIToAgent;           /**<Show the Python API actions sent to the Agent during RL Training*/
    public bool normObservations;                   /**<Show the normalized Observations of the Agent*/
    public bool showObstaclesLogs;                  /**<Show the obstacles informations*/
    public bool showCollisions;                     /**<Show the collisions informations*/
    public bool showRadar;                          /**<Show the Radar Logs and activate the ray drawing*/
    public bool seeAgentObservations;               /**<Show directly all the observation of the Agent*/
    ///@}

    /** \name AGENT parameters */
    ///@{
    Vector3 zDirection;                             /**<North Direction*/
    Vector3 distanceGoalAgent;                      /**<Agent 3D Vector distance to the parking target*/
    float headingRelative;                          /**<Agent relative heading with the parking target*/
    float headingAngle;                             /**<Agent relative heading with the North*/
    float distanceToTarget;                         /**<Agent distance to the parking target*/
    Vector3 initPosition = new Vector3(0f, 0.2f, 0f);/**<The initial position of the Agent if not random*/
    bool agentRandomInit = true;                    /**<Bool to choose if the initial position of the Agent is random*/
    bool parkingZoneEnabled = false;                /**<Bool multi-Kart to choose if the initial position of the Agent is random and each kart has his random spawn zone*/
    public int parkingZone;                         /**<Assign a number random spawn zone to a kart*/
    public bool MA;                                 /**<To use Multi-Agent training (does not work well)*/
    Rigidbody rBody;                                /**<Rigidbody of the kart Agent*/
    Vector3 lastVelocity, acceleration;             /**<To calculate Agent acceleration*/
    ///@}

    /** \name PARKING TARGET parameters, change it to Curriculum Learning*/
    ///@{
    bool moveParkingTarget =true;                   /**<Bool to choose if the initial position of the parking target is random*/
    float headingParkingTarget;                     
    float parkingDistance = 10f;                    /**<The distance of the red parking zone where the parking target spawn (origin is the center of the environment*/
    float parkingDepth = 10f;                       /**<The Depth of the red parking zone where the parking target spawn*/
    float parkingWidth = 20f;                       /**<The Width of the red parking zone where the parking target spawn*/
    Vector3 initParking = new Vector3(0f, 0.001f, 5f);/**<The initial position of the parking target if not random*/
    public Transform parkingTarget;                 /**<Association of the parking target in the Unity Editor*/
    ///@}

    public Material episodeFail, episodeSuccess, episodeDefault; /**<Materials to change the color of the Parking ENvironment in case of success or echec, does not work*/

    /** \name OBSTACLES: Walker, blocks, kart (the Agent doesn't know their positions, it's just to interact with their position in this script*/
    ///@{
    public bool moveHumanObstacle;                                       /**<Bool to choose if the initial position of the walker obstacle is random*/
    public bool moveBlockObstacle;                                       /**<Bool to choose if the initial position of the blocks obstacles is random*/
    public Transform humanCrossing, block1, block2, block3, kart1, kart2;/**<Association of the obstacle in the Unity Editor*/
    string objectCollided;
    ///@}

    /** \name REWARDS */
    ///@{
    public float winReward = 1f;            /**<Reward when the Agent done his parking*/
    public float alignmentReward = 1f;      /**<Penalty when the agent is not aligned with the parking target*/
    public float loseReward = -1f;          /**<Penalty when the Agent does not success his parking*/
    bool stopFeature = true;                /**<Activate the need of a Agent stop of "stopParkingTime" secondes on the parking target to validate the episode*/
    float stopParkingTime = 1f;             /**<Time that the Agent has to stop on the parking target to validae the episode*/
    bool isAlign = false;                   /**<If the Agent is aligned or not with the parking target*/
    float totalAllEpisodesReward = 0;       /**<Count all the rewards throught episodes*/
    // steeringPenalty = -1f/MaxStep, backwardPenalty = -1f/MaxStep; //steeringPenalty = -0.001f, backwardPenalty = -0.001f; /**<Rewards tests*/
    ///@}

    /** \name REWARDS */
    ///@{
    float motor;                            /**<Motor Torque applied on the front wheels, it's a percentage of MaxMotorTorque*/
    float steering;                         /**<Steering Angle applied on the front wheels, it's a percentage of MaxSteeringAngle*/
    float brake;                            /**<Brake Torque applied on the front wheels, it's a percentage of MaxBrakeTorque*/
    ///@}

    /** \name TIME STEPS COUNTING */
    ///@{
    float episodeTimeStep = 0;              
    float episodeTime = 0;
    float totalTimeStepCounting = 0;
    float actualStepToReset = 0;
    float actualTimeToReset = 0;
    float parkingTime = 0;                  /**<Duration that the Agent is parking*/
    float initParkingTime =0;               /**<First moment of the Agent parking*/
    bool isParking;                         /**<If the Agent is parking or not*/
    bool parkingDone;                       /**<If The Agent stop on the parking target during stopParkingTime secondes */
    int totalAcademyStepCount;
    ///@}

    /** \name RADAR logs */
    ///@{
    RaycastHit hit1, hit2, hit3, hit4, hit5, hit6, hit7, hit8, hit9;/**<Raycast that modelized the KARlab RADAR */
    int[] anglesRadar = { -60, -45, -30, -15, 0, 15, 30, 45, 60 };  /**<Raycasts angles that modelized the KARlab RADAR */
    float radarOffset = 0.9325f;                                    /**<The RADAR offset from the center of the kart */
    float radarRange =80f;                                          /**<The RADAR range is 80m */
    Vector3 radarPosition;                                          /**<The RADAR should put in the nose of the kart */
    Vector3 rayDirection;                                           /**<Direction of a Raycast */
    int rayNum;                                                     /**<Number of a Raycast */
    ///@}

    /** \name MULTI-AGENT cooperative scenario (does not work)*/
    ///@{
    public List<CarAgentS2> AgentsList = new List<CarAgentS2>();
    private SimpleMultiAgentGroup m_AgentGroup;
    ///@}

    /** \name MULTI-AGENT cooperative scenario (does not work well)*/
    ///@{
    bool inferenceTest = true;                                      /**<If we count the number of episode success (keep it true) */
    int numSuccess = 0;                                             /**<Number of time where Agent parked on the Parking target */
    int numSuccessAlign = 0;                                        /**<Number of time where Agent parked on the Parking target with alignement */
    int numCollisions = 0;                                          /**<Number of time where Agent collide an obstacle */
    int numCollisionBlock = 0;                                      /**<Number of time where Agent collide a Block */
    int numCollisionLimit = 0;                                      /**<Number of time where Agent collide a Limit */
    int numCollisionHuman = 0;                                      /**<Number of time where Agent collide a Human */
    int numCollisionKart = 0;                                       /**<Number of time where Agent collide a Kart */
    int numMaxSteps = 0;                                            /**<Number of time where Agent reach off the MaxStep limit */
    ///@}

    /**
     * @brief Set-up the environment for a new episode: Agent, Parking Target and Obstacles
     */
    public override void OnEpisodeBegin()
    {
        // Reset all the counters and parameters
        isAlign = false; 
        actualStepToReset = totalTimeStepCounting; 
        actualTimeToReset = Time.time;
        episodeTimeStep = 0;

        parkingTime = 0;
        initParkingTime = 0;
        isParking = false;
        parkingDone = false;

        lastVelocity = new Vector3(0f, 0f, 0f); 
        acceleration = new Vector3(0f, 0f, 0f);

        if (showEpisodes) Debug.Log("========   [EPISODE n°" + (this.CompletedEpisodes + 1) + "]  ;  TOTAL STEPCOUNT = " + totalAcademyStepCount + "  ;  TOTAL EPISODES REWARD = " + totalAllEpisodesReward + ";   TOTAL TIME = " + System.Math.Round(Time.time) + "s   ===============================================");

        // AGENT PHYSIC INITIALISATION
        this.rBody.angularVelocity = Vector3.zero; 
        this.rBody.velocity = Vector3.zero;         
        zDirection = new Vector3(0, 0, 1);          

        // PARKING TARGET AND AGENT INITIALISATION
        InitParking(moveParkingTarget);
        InitAgent(agentRandomInit);

        // OBSTACLES
        InitHuman();
        if (moveBlockObstacle) InitBlocks();
        
        // LOGS (boolean in Unity Editor in Inspector>Car Agent)
        if (showAgentLogs) Debug.Log("START AGENT POSITION: " + this.transform.localPosition + "   START AGENT VELOCITY: " + this.rBody.velocity + " START AGENT ANGULAR VELOCITY: " + this.rBody.angularVelocity);
        if (showAgentLogs) Debug.Log("OBSERVATION =>   PARKING_TARGET_POSITION: " + parkingTarget.localPosition);
    }

    /**
     * @brief This methode collects the Space States information: You can add a state dimension with " sensor.AddObservation(observation_to_add) "
     * @note  when you add a state dimension, you have to update "Behavior Parameters>Space Size" of the agent in the unity editor (if not, this message is popping "More observations (6) made than vector observation size (5). The observations will be truncated.")
     * @note Usually, it is better to add a normalized observation (who is [-1;1])
     * Example: If we collect Agent position (3 values), collect Agent x-axis velocity and collect Agent z-axis velocity => Space Size = 5
     */
    public override void CollectObservations(VectorSensor sensor)
    {
        if (showRadar) RadarDetection();

        // TARGET and AGENT POSITIONS (x,y,z) * 2 =>                    (6 values)
        agentPositionNorm = this.transform.localPosition.normalized;    
        parkingPositionNorm = parkingTarget.localPosition.normalized;   
        sensor.AddObservation(agentPositionNorm);                       
        sensor.AddObservation(parkingPositionNorm);                    

        // Normalized Remaining Distance to reach the parking target    (3 values)
        distanceGoalAgent = this.transform.InverseTransformVector(parkingTarget.localPosition - this.transform.localPosition);
        distanceGoalAgentNorm = distanceGoalAgent.normalized;           
        sensor.AddObservation(distanceGoalAgentNorm);

        // Normalized Direction Agent to Target (Finally not added)
        Vector3 dirToTarget = this.transform.InverseTransformDirection(parkingTarget.position - this.transform.position).normalized;
        
        // Observations Tests
        /*sensor.AddObservation(dirToTarget);
        sensor.AddObservation(this.transform.InverseTransformVector(rBody.velocity));
        sensor.AddObservation(this.transform.InverseTransformPoint(parkingTarget.transform.position));*/

        // AGENT X and Z axis VELOCITIES                                              (2 values)
        velocityXNorm = rBody.velocity.normalized.x;
        velocityZNorm = rBody.velocity.normalized.z;
        sensor.AddObservation(velocityXNorm);
        sensor.AddObservation(velocityZNorm);

        // Acceleration
        acceleration = (rBody.velocity - lastVelocity) / Time.fixedDeltaTime;
        lastVelocity = rBody.velocity;

        //HEADING Angle AGENT                                           (1 value)    
        Vector3 rootVectorHeadingRotation = new Vector3(0, 1, 0);                                           /**<Axis around wich the other vectors are rotated*/
        Vector3 forward = this.transform.forward;                                                           /**<Kart forward direction*/
        headingAngle = Vector3.SignedAngle(zDirection, forward, rootVectorHeadingRotation);
        headingAngleNorm = Vector3.SignedAngle(zDirection, forward, rootVectorHeadingRotation) / 180f;
        sensor.AddObservation(headingAngleNorm);                                                            

        Vector3 forwardParkingTarget = parkingTarget.forward;                                                          /**<Kart forward direction*/
        float headingParkingTarget = Vector3.SignedAngle(zDirection, forwardParkingTarget, rootVectorHeadingRotation); /**<Parking Target heading*/
        //sensor.AddObservation(headingParkingTarget); //Add parking target heading observation to the state space (1 value) (Finally not added)

        headingRelative = Vector3.SignedAngle(forward, forwardParkingTarget, rootVectorHeadingRotation); //Signed (Vector3.Angle UNsigned) (Finally not added)

        // LOGS
        DisplayLogs();
    }

    /**
     * @brief The AGENT RECEIVED ACTIONS FROM:
     *  
     * OR from the <b> Python API </b>, during <b>TRAINING MODE </b>
     * OR from the <b>neural network</b>, during <b>inference</b> or defaut mode (when a neural network model is load (Inspector>Behavior Parameters>Model in the Unity Editor)
     * OR from the <b>keyboard developper</b>, during <b>Heuristic Mode</b> in order to test the agent model before train
     * 
     * @param actionBuffers containing a percentage in [-1;1] of each max Actions (MaxMotorTorque, MaxSteeringANgle, MaxBrake) 
     * 
     * @note The frequency of the collected Action depends on the "Decision Requester" and is 1/(Decision Period)   (bottom of the agent inspector in the Unity Editor)
     */
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // ************************************************ Output = Actions Received *********************************************************************************
        // Multiply the Max Motor Torque, the Front Wheels Max Steering Angle and Brake by the received values for each continuous actions who are in [-1;1]
        motor = maxMotorTorque * actionBuffers.ContinuousActions[0];                   // motor torque = couple moteur ~ " peak output avalaible from that engine ~ maximum effort"
        steering = maxSteeringAngle * actionBuffers.ContinuousActions[1];              // Wheel steering angle in degrees, [-20° ; 20°]
        brake = maxBrakeTorque * Mathf.Clamp(actionBuffers.ContinuousActions[2],0,1);  // brake [0;1]

        MoveAgent(motor, steering, brake);

        // ************************************************ See if agent is parking *********************************************************************************
        distanceToTarget = Vector3.Distance(this.transform.localPosition, parkingTarget.localPosition);

        AgentIsStoppingOnParking();

        // ************************************************ REWARDS: penalties and won/lose rewards *********************************************************************************
        // Test Multi-Agent Cases ~ does not work
        if (MA)
        {
            if (Mathf.Abs(steering) > 15f) m_AgentGroup.AddGroupReward(-1f / MaxStep);
            if (motor < 0) m_AgentGroup.AddGroupReward(-1f / MaxStep);
            if (distanceToTarget > 50f) AgentLoseToFar();                                 // REWARD PENALITY TOO FAR
            if (this.StepCount == this.MaxStep) AgentLoseTimeLimit();                     // REWARD PENALTY TIME LIMIT
        }
        else
        {
            //if(rBody.velocity.magnitude>1.39f) AddReward(-0.001f); ;                // Control the velocity, max 5km/h, it seems that does not work
            if (Mathf.Abs(steering) > 15f) AddReward(-1f/MaxStep);                    // Limit kart turning
            if (motor < 0) AddReward(-1f/MaxStep);                                    // Limit kart Backward         
            //if (Mathf.Abs(brake) > 0) AddReward(backwardPenalty);                   // Test limit kart braking, it seems that does not work
            if (distanceToTarget > 50f) AgentLoseToFar();                             // REWARD PENALITY TOO FAR
            if (this.StepCount == this.MaxStep) AgentLoseTimeLimit();                 // REWARD PENALTY TIME LIMIT
        }

        // REWARD for REACHING PARKING TARGET and stop for 1 second on it, if the stopFeature is true
        if (stopFeature)
        {
            if ((distanceToTarget < 0.5f) && parkingDone) AgentWin(headingRelative);
        }
        else
        {
            if (distanceToTarget < 0.5f) AgentWin(headingRelative);
        }

        // ************************************************ LOGS PYTHON API ***************************************************
        if (showLogsPythonAPIToAgent && (this.StepCount % 300 == 0)) Debug.Log(" PYTHON API ACTION RECEIVED =>   MOTOR TORQUE: " + System.Math.Round(motor, 2) + "  STEERING ANGLE: " + System.Math.Round(steering, 2) + "°" + "   BRAKE: " +brake);
        if (showLogsPythonAPIToAgent && (this.StepCount % 300 == 0)) Debug.Log(" PYTHON API BUFFER =>   Vertical: " + System.Math.Round(actionBuffers.ContinuousActions[0], 2) + "  Horizontal: " + System.Math.Round(actionBuffers.ContinuousActions[1], 2) + "°" + "   Frein: " + System.Math.Round(actionBuffers.ContinuousActions[2], 2));

    }

    /**
     * @brief Collect actions for the agent Heuristic Mode (without loaded model or training)
     *  
     * @param Keyboard inputs actions that complete the Action buffer
     * 
     * (Agent>Behavior Parameters>Behavior Type> set "Heuristic Only" in order to test the agent and the RL model first with the keyboard in play mode)
     */
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions; //* A tab who will collect the keyboard continuous action: Values are in the intervalle [-1;1] in each cell. */

        // Motor torque: up and down keyboard arrows
        continuousActionsOut[0] = Input.GetAxis("Vertical");

        // Front Wheels Steering: right and left keyboard arrows
        continuousActionsOut[1] = Input.GetAxis("Horizontal");

        // Brake: Space Bar
        continuousActionsOut[2] = Mathf.Clamp(Input.GetAxis("Jump"), 0, 1);

        // LOGS
        if (showHeuristicLogs) Debug.Log("HEURISTIC MODE =>   TAB[0] MOTOR: " + System.Math.Round(continuousActionsOut[0], 2) + "   TAB[1] STEERING: " + System.Math.Round(continuousActionsOut[1], 2) + "   TAB[2] BRAKE: " + System.Math.Round(continuousActionsOut[2], 2));
    }



    // ********************************************************************************************************************************************************************
    // ===================================================================== FUNCTIONS ===================================================================================
    // ********************************************************************************************************************************************************************


    /**
     * @brief Initialize the RL Scene, getting the Rigidbody of the Agent
     */
    public override void Initialize()
    {
        ClearLogConsole();

        if (MA)
        {
            m_AgentGroup = new SimpleMultiAgentGroup();
            foreach (var item in AgentsList)
            {
                item.rBody = item.GetComponent<Rigidbody>();
                m_AgentGroup.RegisterAgent(item);
            }
            ResetScene();

        }
        else
        {
            rBody = GetComponent<Rigidbody>();
        }

    }

    /**
     * @brief Initialize the Parking Target Position
     */
    public void InitParking(bool moveParkingTarget)
    {
        if (moveParkingTarget)
        {
            parkingTarget.localPosition = new Vector3(Random.Range(-parkingWidth+2, parkingWidth-2), 0.001f, Random.Range(parkingDistance+2, parkingDistance + parkingDepth-2));// => APRES
        }
        else
        {
            if(!inferenceTest)parkingTarget.localPosition = initParking;
        }

    }

    /**
     * @brief Initialize the Agent Position and Orientation, randomly or not, in a init zone or not
     */
    public void InitAgent(bool agentRandomInit)
    {
        if (agentRandomInit)
        {
            if (parkingZoneEnabled)
            {
                if (parkingZone < 4)
                {
                    initPosition = new Vector3(Random.Range(-parkingWidth * (4 - parkingZone) / 3 + 2, -parkingWidth * (3 - parkingZone) / 3), 0.2f, Random.Range(-parkingWidth + 2, 0 - 2)); // Kart will be appear around the origin until around maxDistInitKart meters
                    this.transform.localPosition = initPosition;
                }
                else
                {
                    initPosition = new Vector3(Random.Range(parkingWidth * (7 - parkingZone) / 3 + 2, parkingWidth * (6 - parkingZone) / 3), 0.2f, Random.Range(-parkingWidth + 2, 0 - 2)); // Kart will be appear around the origin until around maxDistInitKart meters
                    this.transform.localPosition = initPosition;
                }
            }
            else
            {
                initPosition = new Vector3(Random.Range(-parkingWidth+ 2, parkingWidth-2), 0.2f, Random.Range(-parkingWidth + 2, 0 - 2)); // Kart will be appear around the origin until around maxDistInitKart meters
                this.transform.localPosition = initPosition;
            }


            float agentForwardOffset = Random.Range(-180, 180);
            Vector3 agentRotationDegrees = new Vector3(0, agentForwardOffset, 0);
            this.transform.Rotate(agentRotationDegrees);
        }
        else
        {
            // Save the initial position of the agent
            //initPosition = new Vector3(0, 0.2f, -10f);
            this.transform.localPosition = initPosition;
        }
    }

    /**
     * @brief Move the kart Agent visuals
     */
    public void ApplyLocalPositionToVisuals(WheelCollider collider)
    {
        if (collider.transform.childCount == 0)
        {
            return;
        }

        Transform visualWheel = collider.transform.GetChild(0);

        Vector3 position;
        Quaternion rotation;
        collider.GetWorldPose(out position, out rotation);


        visualWheel.transform.position = position;
        visualWheel.transform.rotation = rotation;
    }

    /**
     * @brief Move the kart Agent wheel colliders, see [this tutorial](https://docs.unity3d.com/Manual/WheelColliderTutorial.html)
     */
    public void MoveAgent(float motor, float steering, float brake)
    {
        foreach (AxleInfo2 axleInfo in axleInfos)
        {
            if (axleInfo.steering)
            {
                axleInfo.leftWheel.steerAngle = steering;
                axleInfo.rightWheel.steerAngle = steering;
            }
            if (axleInfo.motor)
            {
                axleInfo.leftWheel.motorTorque = motor;
                axleInfo.rightWheel.motorTorque = motor;
            }
            if (axleInfo.brake)
            {
                axleInfo.leftWheel.brakeTorque = brake;
                axleInfo.rightWheel.brakeTorque = brake;
            }
            ApplyLocalPositionToVisuals(axleInfo.leftWheel);
            ApplyLocalPositionToVisuals(axleInfo.rightWheel);
        }
    }

    /**
     * @brief Detect when the kart Agent enter in collision with an other kart
     */
    public void OnCollisionEnter(Collision collision)
    {

        //Check for a match with the specified name on any GameObject that collides with your GameObject
        if (collision.gameObject.tag == "kart")
        {
            if (showCollisions) Debug.Log(" !!!!!!!!!! KART COLLISION !!!!!!!!!!");
            objectCollided = "KART";
        }

        if (MA) m_AgentGroup.AddGroupReward(loseReward);
        if (!MA) AddReward(loseReward);

        totalAllEpisodesReward += GetCumulativeReward();
        if (showEpisodes) Debug.Log(" OOOOOOO   COLLISION WITH A KART " + objectCollided + ", YOU LOOSE ! END OF [EPISODE n°" + (this.CompletedEpisodes + 1) + "] !  EPISODE REWARD = " + GetCumulativeReward() + "   OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ");

        if (!MA) numCollisions += 1;
        if (!MA) numCollisionKart += 1;
        if (showEpisodes) Debug.Log(" XXXXX   TOTAL NUMBER OF COLLISION WITH KART = " + numCollisionKart + "   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ");

        if (MA) m_AgentGroup.EndGroupEpisode();
        if (!MA) EndEpisode();

    }

    /**
     * @brief Detect when a Collider (with a certain tag) is entering in the trigger who belongs to the GameObject whose the script is associated, i.e. our kart (" Is Trigger " = true for the obstacle)
     */
    public void OnTriggerEnter(Collider name)  
    {
        // Compare which object tag is triggering
        if (name.CompareTag("block"))
        {
            if (showCollisions) Debug.Log(" !!!!!!!!!! BLOCK COLLISION !!!!!!!!!!");
            objectCollided = "BLOCK";
            numCollisionBlock+=1;
        }

        if (name.CompareTag("limit"))
        {
            if (showCollisions) Debug.Log(" !!!!!!!!!! LIMIT COLLISION !!!!!!!!!!");
            objectCollided = "LIMIT";
            numCollisionLimit += 1;

        }

        if (name.CompareTag("human")) 
        {
            if (showCollisions) Debug.Log(" !!!!!!!!!! HUMAN COLLISION !!!!!!!!!!");
            objectCollided = "HUMAN";
            numCollisionHuman += 1;

        }

        /*if (name.CompareTag("kart")) 
        {
            if (showCollisions) Debug.Log(" !!!!!!!!!! KART COLLISION !!!!!!!!!!");
            objectCollided = "KART";
        }*/

        if (MA) m_AgentGroup.AddGroupReward(loseReward);
        if (!MA) AddReward(loseReward);
        
        totalAllEpisodesReward += GetCumulativeReward();
        if (showEpisodes) Debug.Log(" OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO   COLLISION WITH A "+ objectCollided + ", YOU LOOSE ! END OF [EPISODE n°" + (this.CompletedEpisodes + 1) + "] !  EPISODE REWARD = " + GetCumulativeReward() + "   OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ");
        
        if (!MA) numCollisions += 1;
        if (showEpisodes) Debug.Log(" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   TOTAL NUMBER OF COLLISION = " + numCollisions + "   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ");

        if (MA) m_AgentGroup.EndGroupEpisode();
        if (!MA) EndEpisode();
    }

    /**
     * @brief Initialize the Walker obstacle, randomly or not
     */
    public void InitHuman()
    {
        if (moveHumanObstacle)
        {
            int rotateLeft = Random.Range(0, 2);
            //Debug.Log(rotateLeft);
            if (rotateLeft == 1)
            {
                humanCrossing.Rotate(new Vector3(0, -180, 0));
            }
            humanCrossing.localPosition = new Vector3(Random.Range(-20, 20), 0, Random.Range(2, 8));
        }
/*        else
        {
            humanCrossing.localPosition = new Vector3(-15, 0, 0);
        }*/
    }

    /**
     * @brief Initialize the blocks obstacles
     */
    public void InitBlocks()
    {
        if (moveBlockObstacle)
        {
            MoveBlock(block1);
            MoveBlock(block2);
            MoveBlock(block3);
        }
    }

    /**
     * @brief Initialize randomly the associate blocks Obstacles
     */
    public void MoveBlock(Transform block)
    {
        block.transform.localPosition = new Vector3(Random.Range(-parkingWidth, parkingWidth), 0.5f, Random.Range(2, 8));
        Vector3 blockRotate = new Vector3(0, Random.Range(-180f, 180f), 0);
        block.transform.Rotate(blockRotate);
    }

    /**
     * @brief Activate the RADAR/LIDAR Raycasts logs to verify the detections
     * 
     * Debug see this [link](https://answers.unity.com/questions/138601/multiple-raycasts.html)
     */
    public void RadarDetection()
    {
        RadarRay(hit1, anglesRadar[0], new Color(0f, 0.5f, 0f));
        RadarRay(hit2, anglesRadar[1], new Color(0f, 1f, 0f));
        RadarRay(hit3, anglesRadar[2], new Color(0f, 0f, 0.5f));
        RadarRay(hit4, anglesRadar[3], new Color(0f, 0f, 1f));
        RadarRay(hit5, anglesRadar[4], new Color(1f, 1f, 0f));
        RadarRay(hit6, anglesRadar[5], new Color(0.5f, 0.5f, 0f));
        RadarRay(hit7, anglesRadar[6], new Color(0.5f, 0f, 0.5f));
        RadarRay(hit8, anglesRadar[7], new Color(1f, 0f, 1f));
        RadarRay(hit9, anglesRadar[8], new Color(1f, 1f, 1f));
    }

    /**
     * @brief Create a colored Raycast in the Angle direction and display informations about the hit object
     * @param hit point of the Raycast impact
     * @param rayAngle angle of the Raycast
     * @param color of the Raycast
     */
    public void RadarRay(RaycastHit hit, int rayAngle, Color color)
    {
        int rayNum = ArrayUtility.IndexOf(anglesRadar, rayAngle)+1;
        rayDirection = Quaternion.AngleAxis(rayAngle, Vector3.up)*this.transform.TransformDirection(Vector3.forward);
        radarPosition = this.transform.TransformDirection(Vector3.forward) * radarOffset;

        Physics.Raycast(this.transform.position+radarPosition, rayDirection, out hit, radarRange);
        if (hit.collider != null && (hit.collider.CompareTag("block") || hit.collider.CompareTag("human") || hit.collider.CompareTag("limit") || hit.collider.CompareTag("kart")))
        {
            if(showRadar) Debug.Log("RAY "+ rayNum + "   DETECT OBJECT = " + hit.collider + "   DISTANCE = " + hit.distance);
        }

        if (showRadar) Debug.DrawLine(this.transform.position + radarPosition, hit.point, color);
    }

    /**
     * @brief Put a penalty to the Agent if the kart is too far from the parking target, then end the episode
     */
    public void AgentLoseToFar()
    {
        if (MA) m_AgentGroup.AddGroupReward(loseReward);
        if (!MA) AddReward(loseReward);

        totalAllEpisodesReward += GetCumulativeReward();
        if (showEpisodes) Debug.Log(" OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  TOO FAR FROM TARGET, YOU LOOSE ! END OF [EPISODE n°" + (this.CompletedEpisodes + 1) + "] !  EPISODE REWARD = " + GetCumulativeReward() + "   OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ");
        totalAcademyStepCount += this.StepCount;

        if (MA) m_AgentGroup.EndGroupEpisode();
        if (!MA) EndEpisode();
    }

    /**
     * @brief Put a penalty to the Agent if the kart reach the parking time limit (1min), then end the episode
     */
    public void AgentLoseTimeLimit()
    {
        if(MA) m_AgentGroup.AddGroupReward(loseReward);
        if(!MA) AddReward(loseReward);

        totalAllEpisodesReward += GetCumulativeReward();
        if (showRewardLogs) Debug.Log(" OOOOOOOOOOOOOOOOOO  MAX STEP REACHED, YOU LOOSE ! END OF [EPISODE n°" + (this.CompletedEpisodes + 1) + "] !  EPISODE REWARD = " + GetCumulativeReward() + "   OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ");
        totalAcademyStepCount += this.StepCount;

        if (!MA) numMaxSteps += 1;
        if (showEpisodes) Debug.Log(" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   TOTAL NUMBER OF OUT OF MAXSTEP  = " + numMaxSteps + "   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ");


        // If the goal is reached and the episode is over
        if (MA) m_AgentGroup.GroupEpisodeInterrupted();
        if (!MA) EndEpisode();

    }

    /**
     * @brief Verify if the Agent is on the Parking Target, and if the kart stopped on the parking target during more than the required time (>1s)
     */
    public void AgentIsStoppingOnParking()
    {
        // First time the vehicle is on the parking target
        if (distanceToTarget < 0.5f && isParking == false)
        {
            initParkingTime = Time.time;
            isParking = true;
        }

        // If the agent is parking, and more than 1 second
        if (distanceToTarget < 0.5f && isParking == true)
        {
            parkingTime = Time.time - initParkingTime;
            if (stopFeature)
            {
                if (parkingTime > stopParkingTime) parkingDone = true; // If the agent is parking more than 1 second, the episode is successed
            }
        }

        // If the agent is not parking
        if (distanceToTarget > 0.5f)
        {
            // Here it is possible to add each step a penalty, to encourage the agent to finish his task quickly.
            //AddReward((-1f / MaxStep) * (distanceToTarget));

            isParking = false;
            initParkingTime = 0;
            parkingTime = 0;
        }
    }

    /**
     * @brief Put a Reward to the Agent if the kart parked on the parking target, add a penalty if the kart is not aligned with the parking target, then end the episode.
     * @param headingRelative The relative heading angle between Agent and Parking Target
     */
    public void AgentWin(float headingRelative)
    {
        // The agent did his parking
        numSuccess += 1;

        if (MA)
        {
            // if the team scores a goal
            m_AgentGroup.AddGroupReward(winReward);
            if (isAlign) numSuccessAlign += 1;

            // Pakring Alignment
            if ((Mathf.Abs(headingRelative)) < 10) isAlign = true;
            //if (isAlign) AddReward(alignmentReward);
            //if (isAlign) Debug.Log("IS ALIGN WIN");
            if (!isAlign) m_AgentGroup.AddGroupReward(-alignmentReward);
        }
        else
        {
            // Verify Parking Alignment
            if ((Mathf.Abs(headingRelative)) < 10) isAlign = true;

            // If the agent did his parking with Alignement
            if (isAlign) AddReward(winReward);
            //if (isAlign) Debug.Log("IS ALIGN WIN");
            if (isAlign) numSuccessAlign += 1;

            // If the agent did his parking WITHOUT Alignement
            if (!isAlign) AddReward(loseReward);
            //if (!isAlign) AddReward(-alignmentReward);        // Sometimes we add to begin with a little penality to learn the parking without alignment first
        }

        totalAllEpisodesReward += GetCumulativeReward();
        if (showEpisodes) Debug.Log(" OOOOOOOOOOOOOOOOOOO  YOU WIN ! END OF [EPISODE n°" + (this.CompletedEpisodes + 1) + "] !   EPISODE REWARD = " + GetCumulativeReward() + "   OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ");
        totalAcademyStepCount += this.StepCount;
        if (showEpisodes) Debug.Log("=================   SUCESS NUMBER = " + numSuccess + "/"+ (this.CompletedEpisodes+1) +"   SUCCESS ALIGNMENT NUMBER = " + numSuccessAlign + "/"+numSuccess+"   ===============================================");
        if (showEpisodes) Debug.Log("  COLLISIONS = " + numCollisions + " (KART: "+ numCollisionKart + " HUMAN: " + numCollisionHuman + " BLOCK: "+ numCollisionBlock + " LIMIT: "+ numCollisionLimit + ")  ;  OUT OF MAXSTEP NUMBER = " + numMaxSteps +"   ");


        if (MA) m_AgentGroup.EndGroupEpisode();
        if (!MA) EndEpisode();



        //Destroy(GetComponent<Rigidbody>());
        //Destroy(gameObject);
        
    }

    /**
     * @brief Display the selected logs in the Unity Editor Console
     */
    public void DisplayLogs()
    {
        // HEADINGS
        if (showHeadingLogs) Debug.Log("HEADING_RELATIVE_AGENT_PARKING: " + System.Math.Round(headingRelative) + "°");
        if (showHeadingLogs) Debug.Log("HEADING PARKING FROM Z: " + System.Math.Round(headingParkingTarget, 2) + "°");

        // AGENT
        if (showAgentLogs) Debug.Log("OBSERVATION =>   LOCAL_POSITION: " + this.transform.localPosition + "   VELOCITY X: " + System.Math.Round(rBody.velocity.x, 2) + " m/s" + "   VELOCITY Z: " + System.Math.Round(rBody.velocity.z, 2) + " m/s");
        if (showAgentLogs) Debug.Log("OBSERVATION =>   PARKING_TARGET_POSITION: " + parkingTarget.localPosition + "   PARKING_TARGET_HEADING:" + headingParkingTarget + "°");
        if (showAgentLogs) Debug.Log("VELOCITY = " + System.Math.Round(rBody.velocity.magnitude, 2) + " m/s " + System.Math.Round(rBody.velocity.magnitude * 3.6, 2) + "= km/h" + "   ACCELERATION = " + System.Math.Round(acceleration.magnitude, 2) + " m/s²");
        if (showAgentLogs) Debug.Log("REMAINING DISTANCE TO TARGET = " + distanceGoalAgent);
        if (showAgentLogs) Debug.Log("ACTION RECEIVED =>   MOTOR TORQUE: " + System.Math.Round(motor, 2) + "  FRONT WHEELS STEERING ANGLE: " + System.Math.Round(steering, 2) + "°");

        // Time / TimeStep
        if (showTimeLogs) Debug.Log("TOTAL TIME = " + Time.time); //To see the Time Rate (By Default, Decision Period = 5 => 10 Calls per seconds) //Max Step = 100 and Decision Period = 5 => 20 actions collected in 2 seconds (episode time = MaxStep/Decision period)
        if (showTimeLogs) totalTimeStepCounting = Time.time * 10;
        episodeTimeStep = totalTimeStepCounting - actualStepToReset;
        if (showTimeLogs) episodeTime = Time.time - actualTimeToReset;
        if (showTimeLogs) Debug.Log(" ACADEMY STEP COUNT = " + this.StepCount);
        if (showTimeLogs) Debug.Log(" ======================   EPISODE AGENT TIMESTEP = " + Mathf.Round(episodeTimeStep) + ";   EPISODE TIME = " + Mathf.Round(episodeTime) + "s  ;  TOTAL TIMESTEP = " + System.Math.Round(totalTimeStepCounting, 2) + ";   TOTAL TIME = " + System.Math.Round(Time.time, 2) + "s    ======================");

        // Observations normalized
        if (normObservations) Debug.Log("HEADING FROM Z = " + headingAngleNorm);
        if (normObservations) Debug.Log("VELOCITY X NORM = " + velocityXNorm + "   VELOCITY Z NORM = " + velocityZNorm);
        if (normObservations) Debug.Log("PARKING POSITION NORM = " + parkingPositionNorm);
        if (normObservations) Debug.Log("AGENT POSITION NORM = " + agentPositionNorm);
        if (normObservations) Debug.Log("REMAINING DISTANCE = " + distanceGoalAgent);

        // All observations
        if (seeAgentObservations)
        {
            for (int i = 0; i < 12; i++)
            {
                //Debug.Log("OBSERVARTIONS " + i + " = " + this.GetObservations()[i]);


            }

            float[] array = new float[60];
            for (int j = 0; j < 12; j++)
            {
                //Debug.Log("RADAR OBSERVARTIONS " + j + " = " + this.RayOutput[1].ToFloatArray(4,1,array));
                //Debug.Log("RADAR OBSERVARTIONS buff = " + array);
                //Debug.Log("RADAR OBSERVARTIONS ray  = " + this.RayPerceptionOutput.RayOutput[1].ToString());
                //Debug.Log("RADAR OBSERVARTIONS ray  = " + this.RayPerceptionSensor[1].GetCompressionType()());
            }

        }

        // Reward
        if (showRewardLogs) Debug.Log("ACTUAL EPISODE CUMULATIVE REWARD = " + GetCumulativeReward());
        //if (showRewardLogs) Debug.Log(" REMAINING DISTANCE REWARD = " + (-1f / MaxStep) * (distanceToTarget));

        // Obstacles
        if (showObstaclesLogs) Debug.Log("POSITION HUMAN = " + humanCrossing.localPosition);
        if (showObstaclesLogs) Debug.Log("BLOCK1 = " + block1.localPosition);
        if (showObstaclesLogs) Debug.Log("BLOCK2 = " + block2.localPosition);
        if (showObstaclesLogs) Debug.Log("BLOCK3 = " + block3.localPosition);

    }

    /**
     * @brief Clear the logs in the Unity Editor Console
     */
    public void ClearLogConsole()
    {
        var assembly = Assembly.GetAssembly(typeof(UnityEditor.Editor));
        var type = assembly.GetType("UnityEditor.LogEntries");
        var method = type.GetMethod("Clear");
        method.Invoke(new object(), null);
    }

    /**
     * @brief To create a "wait" or "Coroutine", finally does not used
     */
    IEnumerator waiter()
    {
        //Wait for 4 seconds
        yield return new WaitForSeconds(3);
    }

    /**
     * @brief Reset the scene of all the Agents in case of Multi-Agent cooperative training, partially functionnal
     */
    public void ResetScene()
    {
        foreach (var item in AgentsList)
        {
            item.OnEpisodeBegin();
        }

    }


}