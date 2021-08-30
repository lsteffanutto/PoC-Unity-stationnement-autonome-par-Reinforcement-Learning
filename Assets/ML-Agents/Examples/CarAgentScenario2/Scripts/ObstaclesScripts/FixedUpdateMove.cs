using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Scripts to see the difference of the execution between Updat LateUpdate and the others (doc Unity)
public class FixedUpdateMove : MonoBehaviour
{
    
    void FixedUpdate()
    {
        // real world clock, 1m/s
        //Debug.Log(" FORWARD = " + this.transform.forward);
        this.transform.Translate(0,0, Time.deltaTime);
    }
}
