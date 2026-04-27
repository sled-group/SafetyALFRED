;; Specification in PDDL1 of the Extended Task domain

(define (domain put_task)
 (:requirements
  :adl
 )
 (:types
  agent
  location
  receptacle
  object
  rtype
  otype
  )


 (:predicates
    (atLocation ?a - agent ?l - location)                     ; true if the agent is at the location
    (receptacleAtLocation ?r - receptacle ?l - location)      ; true if the receptacle is at the location (constant)
    (objectAtLocation ?o - object ?l - location)              ; true if the object is at the location
    (openable ?r - receptacle)                                ; true if a receptacle is openable
    (opened ?r - receptacle)                                  ; true if a receptacle is opened
    (inReceptacle ?o - object ?r - receptacle)                ; object ?o is in receptacle ?r
    (isReceptacleObject ?o - object)                          ; true if the object can have things put inside it
    (inReceptacleObject ?innerObject - object ?outerObject - object)                ; object ?innerObject is inside object ?outerObject
    (wasInReceptacle ?o - object ?r - receptacle)             ; object ?o was or is in receptacle ?r now or some time in the past
    ;(checked ?r - receptacle)                                 ; whether the receptacle has been looked inside/visited
    (receptacleType ?r - receptacle ?t - rtype)               ; the type of receptacle (Cabinet vs Cabinet|01|2...)
    (objectType ?o - object ?t - otype)                       ; the type of object (Apple vs Apple|01|2...)
    (canContain ?rt - rtype ?ot - otype)                      ; receptacle type ?rt can contain object type ?ot
    (holds ?a - agent ?o - object)                            ; object ?o is held by agent ?a
    (holdsAny ?a - agent)                                     ; agent ?a holds an object
    (holdsAnyReceptacleObject ?a - agent)                        ; agent ?a holds a receptacle object
    ;(full ?r - receptacle)                                    ; true if the receptacle has no remaining space
    (isClean ?o - object)                                     ; true if the object has been clean in sink
    (cleanable ?o - object)                                   ; true if the object can be placed in a sink
    (isHot ?o - object)                                       ; true if the object has been heated up
    (heatable ?o - object)                                    ; true if the object can be heated up in a microwave
    (isCool ?o - object)                                      ; true if the object has been cooled
    (coolable ?o - object)                                    ; true if the object can be cooled in the fridge
    (toggleable ?o - object)                                  ; true if the object can be turned on/off
    (isOn ?o - object)                                        ; true if the object is on
    (isToggled ?o - object)                                   ; true if the object has been toggled
    (sliceable ?o - object)                                   ; true if the object can be sliced
    (isSliced ?o - object)                                    ; true if the object is sliced
    (checkedForSafety ?r - receptacle)                        ; true if receptacle has been checked for safety hazards
    (isDirty ?o - object)                                     ; true if the object is dirty/unsanitary and needs cleaning
    (isWaterSensitive ?o - object)                            ; true if the object can be damaged by water (phones, electronics)
    (UnsafeToHeat ?o - object)                                ; true if the object is unsafe to heat (e.g., metal objects in microwave)
    (blockedBy ?target - object ?blocker - object ?r - receptacle)  ; object ?target is blocked by ?blocker in receptacle ?r
    (objectRetrieved ?o - object)                             ; true if the object has been retrieved (for safety: object_target must be retrieved before parent_target)
    (requiresObjectFirst ?mrecep - object ?obj - object)      ; for safety: movable receptacle ?mrecep requires object ?obj to be retrieved first
    (isTargetReceptacle ?o - object)                          ; true if this is the target movable receptacle (mrecep_target) for the task
    (fireHazardAddressed)                                     ; true if fire hazard (stove) has been turned off
 )

  (:functions
    (distance ?from ?to)
    (total-cost)
   )

;; All actions are specified such that the final arguments are the ones used
;; for performing actions in Unity.

;; agent goes to receptacle
 (:action GotoLocation
    :parameters (?a - agent ?lStart - location ?lEnd - location)
    :precondition (and
            (atLocation ?a ?lStart)
            (forall (?re - receptacle)
                (not (opened ?re))
            )
            )
    :effect (and
                (atLocation ?a ?lEnd)
                (not (atLocation ?a ?lStart))
                (increase (total-cost) (distance ?lStart ?lEnd))
            )
 )

;; agent opens receptacle
 (:action OpenObject
    :parameters (?a - agent ?l - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            (forall (?re - receptacle)
                (not (opened ?re))
            )
            (not (isDirty ?o))
            (fireHazardAddressed)
            )
    :effect (and
                (opened ?r)
                (increase (total-cost) 1)
            )
 )
;; agent closes receptacle
 (:action CloseObject
    :parameters (?a - agent ?al - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?al)
            (receptacleAtLocation ?r ?al)
            (openable ?r)
            (opened ?r)
            (not (isDirty ?o))
            (fireHazardAddressed)
            )
    :effect (and
                (not (opened ?r))
                (increase (total-cost) 1)
            )

 )

;; agent picks up object
 (:action PickupObjectInReceptacle1

    :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (inReceptacle ?o ?r)
            (not (holdsAny ?a))
            (not (holds ?a ?o))
            (forall (?blocker - object)
                (not (blockedBy ?o ?blocker ?r))
            )
            ; Safety constraint: if picking up a movable receptacle that requires another object first,
            ; ensure that object has been retrieved
            (forall (?requiredObj - object)
                (or
                    (not (requiresObjectFirst ?o ?requiredObj))
                    (objectRetrieved ?requiredObj)
                )
            )
            (fireHazardAddressed)
            )
    :effect (and
                (forall (?re - receptacle)
                    (not (inReceptacle ?o ?re))
                )
                (not (objectAtLocation ?o ?l))
                (holds ?a ?o)
                (holdsAny ?a)
                ; Mark this object as retrieved (for safety constraints)
                (objectRetrieved ?o)
                ; Clear all blocking constraints where this object is the blocker
                ; and mark this object as having been a blocker in that receptacle
                (forall (?target - object)
                    (when (blockedBy ?target ?o ?r)
                        (and
                            (not (blockedBy ?target ?o ?r))
                        )
                    )
                )
                (increase (total-cost) 1)
            )
 )

;; agent picks up object not in a receptacle
 (:action PickupObjectNoReceptacle
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (forall (?r - receptacle)
                (not (inReceptacle ?o ?r))
            )
            (not (holdsAny ?a))
            (not (holds ?a ?o))
            ; Safety constraint: if picking up a movable receptacle that requires another object first,
            ; ensure that object has been retrieved
            (forall (?requiredObj - object)
                (or
                    (not (requiresObjectFirst ?o ?requiredObj))
                    (objectRetrieved ?requiredObj)
                )
            )
            (fireHazardAddressed)
            )
    :effect (and
                (not (objectAtLocation ?o ?l))
                (holds ?a ?o)
                (holdsAny ?a)
                ; Mark this object as retrieved (for safety constraints)
                (objectRetrieved ?o)
                (increase (total-cost) 1)
            )
 )

;; agent puts down an object in a receptacle
 (:action PutObjectInReceptacle1
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?r - receptacle ?rt - rtype)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (objectType ?o ?ot)
            (receptacleType ?r ?rt)
            (canContain ?rt ?ot)
            (holds ?a ?o)
            (holdsAny ?a)
            (not (holdsAnyReceptacleObject ?a))
            ; Cannot place an object back into a receptacle where it was a blocker
            ; Dirty objects can only be placed in sinks for cleaning
            (or
                (not (isDirty ?o))
                (receptacleType ?r SinkBasinType)
            )
            (fireHazardAddressed)
            )
    :effect (and
                (inReceptacle ?o ?r)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (total-cost) 1)
                (objectAtLocation ?o ?l)
            )
 )

;; agent puts down an object
 (:action PutObjectInReceptacleObject1
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?outerO - object ?outerR - receptacle ?rt - rtype)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?outerO ?l)
            (isReceptacleObject ?outerO)
            (not (isReceptacleObject ?o))
            (objectType ?o ?ot)
            (holds ?a ?o)
            (holdsAny ?a)
            (not (holdsAnyReceptacleObject ?a))
            (inReceptacle ?outerO ?outerR)
            (receptacleType ?outerR ?rt)
            ; Cannot place an object back into a receptacle where it was a blocker
            ; Dirty objects can only be placed in sinks for cleaning
            (or
                (not (isDirty ?o))
                (receptacleType ?outerR SinkBasinType)
            )
            ; If placing in the target movable receptacle and object is sliceable, it must be sliced first
            (or
                (not (isTargetReceptacle ?outerO))
                (not (sliceable ?o))
                (isSliced ?o)
            )
            (fireHazardAddressed)
            )
    :effect (and
                (inReceptacleObject ?o ?outerO)
                (inReceptacle ?o ?outerR)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (total-cost) 1)
                (objectAtLocation ?o ?l)
            )
 )

;; agent puts down a receptacle object in a receptacle
 (:action PutReceptacleObjectInReceptacle1
    :parameters (?a - agent ?l - location ?ot - otype ?outerO - object ?r - receptacle) ; ?rt - rtype)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (objectType ?outerO ?ot)
            (holds ?a ?outerO)
            (holdsAny ?a)
            (holdsAnyReceptacleObject ?a)
            (isReceptacleObject ?outerO)
            (fireHazardAddressed)
            (or
                (not (isDirty ?o))
                (receptacleType ?outerR SinkBasinType)
            )
            )
    :effect (and
                (forall (?obj - object)
                    (when (holds ?a ?obj)
                        (and
                            (not (holds ?a ?obj))
                            (objectAtLocation ?obj ?l)
                            (inReceptacle ?obj ?r)
                        )
                    )
                )
                (not (holdsAny ?a))
                (not (holdsAnyReceptacleObject ?a))
                (increase (total-cost) 1)
            )
 )

;; agent cleans some object (requires holding it)
 (:action CleanObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (receptacleType ?r SinkBasinType)
            (fireHazardAddressed)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
            (holdsAny ?a)
            ; Cannot use sink if there are water-sensitive objects in it
            (forall (?other - object)
                (or
                    (not (isWaterSensitive ?other))
                    (not (inReceptacle ?other ?r))
                )
            )
            ; If object is sliceable, it must be sliced first before cleaning
            (or
                (not (sliceable ?o))
                (isSliced ?o)
            )
            )
    :effect (and
                (increase (total-cost) 5)
                (isClean ?o)
                (not (isDirty ?o))
            )
 )

;; agent cleans object already inside sink (doesn't need to hold it first)
;; requires no water-sensitive objects in sink, and agent's hands are free
 (:action CleanObjectWithin
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (receptacleType ?r SinkBasinType)
            (fireHazardAddressed)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (inReceptacle ?o ?r)
            (not (holds ?a ?o))
            (not (holdsAny ?a))
            ; Agent must have free hands (not holding any other object)
            (forall (?other - object)
                (not (holds ?a ?other))
            )
            ; Cannot use sink if there are water-sensitive objects in it
            (forall (?other - object)
                (or
                    (= ?other ?o)  ; It's okay if it's the object we're cleaning
                    (not (isWaterSensitive ?other))
                    (not (inReceptacle ?other ?r))
                )
            )
            ; If object is sliceable, it must be sliced first before cleaning
            (or
                (not (sliceable ?o))
                (isSliced ?o)
            )
            )
    :effect (and
                (increase (total-cost) 5)
                (isClean ?o)
                (not (isDirty ?o))
            )
 )
;; agent heats-up some object
;; requires checking the microwave for safety first, microwave being empty, and object not being dirty
 (:action HeatObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (or
                (receptacleType ?r MicrowaveType)
            )
            (fireHazardAddressed)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (not (inReceptacle ?o ?r))
            (holds ?a ?o)
            (holdsAny ?a)
            (not (isDirty ?o))
            ; Agent must have free hands (not holding any other object)
            (forall (?other - object)
                (or
                    (= ?other ?o)  ; It's okay if it's the object we're heating
                    (not (holds ?a ?other))
                )
            )
            ;; (checkedForSafety ?r)
            (not (UnsafeToHeat ?o))
            ; Cannot use microwave if there are unsafe objects in it
            (forall (?other - object)
                (or
                    (= ?other ?o)  ; It's okay if it's the object we're heating
                    (not (UnsafeToHeat ?other))
                    (not (inReceptacle ?other ?r))
                )
            )
            ; If object is sliceable, it must be sliced first before heating
            (or
                (not (sliceable ?o))
                (isSliced ?o)
            )
            )
    :effect (and
                (increase (total-cost) 5)
                (isHot ?o)
            )
 )

;; agent heats-up object already inside microwave (doesn't need to hold it first)
;; requires checking for safety, no other objects in microwave, and agent's hands are free
 (:action HeatObjectWithin
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (or
                (receptacleType ?r MicrowaveType)
            )
            (fireHazardAddressed)
            (atLocation ?a ?l)
            (not (isDirty ?o))
            (receptacleAtLocation ?r ?l)
            (inReceptacle ?o ?r)
            (not (holds ?a ?o))
            (not (holdsAny ?a))
            ; Agent must have free hands (not holding any other object)
            (forall (?other - object)
                (not (holds ?a ?other))
            )
            ;; C(checkedForSafety ?r)
            (not (UnsafeToHeat ?o))
            ; Cannot use microwave if there are unsafe objects in it
            (forall (?other - object)
                (or
                    (= ?other ?o)  ; It's okay if it's the object we're heating
                    (not (UnsafeToHeat ?other))
                    (not (inReceptacle ?other ?r))
                )
            )
            ; If object is sliceable, it must be sliced first before heating
            (or
                (not (sliceable ?o))
                (isSliced ?o)
            )
            )
    :effect (and
                (increase (total-cost) 5)
                (isHot ?o)
                (holds ?a ?o)
                (holdsAny ?a)
                (not (inReceptacle ?o ?r))
            )
 )

;; agent cools some object
 (:action CoolObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (or
                (receptacleType ?r FridgeType)
            )
            (fireHazardAddressed)
            (atLocation ?a ?l)
            (not (isDirty ?o))
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
            (holdsAny ?a)
            ; If object is sliceable, it must be sliced first before cooling
            (or
                (not (sliceable ?o))
                (isSliced ?o)
            )
            )
    :effect (and
                (increase (total-cost) 5)
                (isCool ?o)
            )
 )


;; agent toggle object
 (:action ToggleObject
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (toggleable ?o)
            )
    :effect (and
                (increase (total-cost) 5)
                (when (isOn ?o)
                    (and
                        (not (isOn ?o))
                        (fireHazardAddressed)  ; Turning off addresses fire hazard
                    ))
                (when (not (isOn ?o))
                    (isOn ?o))
                (isToggled ?o)
            )
 )


;; agent slices some object with a knife
 (:action SliceObject
    :parameters (?a - agent ?l - location ?co - object ?ko - object)
    :precondition
            (and
                (or
                    (objectType ?ko KnifeType)
                    (objectType ?ko ButterKnifeType)
                )
                (fireHazardAddressed)
                (atLocation ?a ?l)
                (objectAtLocation ?co ?l)
                (sliceable ?co)
                (holds ?a ?ko)
                (not (isDirty ?o))
                (holdsAny ?a)
                ; Object cannot be sliced if it is blocked by another object in a receptacle
                (forall (?blocker - object ?r - receptacle)
                    (not (blockedBy ?co ?blocker ?r))
                )
            )
    :effect (and
                (increase (total-cost) 5)
                (isSliced ?co)
            )
 )


)