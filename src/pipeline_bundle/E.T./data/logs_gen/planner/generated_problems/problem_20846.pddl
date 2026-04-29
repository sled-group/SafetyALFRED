
(define (problem plan_20846)
    (:domain put_task)
    (:metric minimize (totalCost))
    (:objects
        agent1 - agent
        Spatula - object
        KeyChain - object
        Fork - object
        Tomato - object
        CD - object
        Vase - object
        ButterKnife - object
        ScrubBrush - object
        Towel - object
        TennisRacket - object
        ShowerGlass - object
        Bread - object
        Cup - object
        Footstool - object
        Newspaper - object
        SoapBottle - object
        Bathtub - object
        Knife - object
        PepperShaker - object
        Statue - object
        Mug - object
        Glassbottle - object
        AlarmClock - object
        BaseballBat - object
        HousePlant - object
        WineBottle - object
        DeskLamp - object
        Television - object
        Potato - object
        WateringCan - object
        Poster - object
        SaltShaker - object
        Pan - object
        Pillow - object
        Plunger - object
        CellPhone - object
        Candle - object
        Kettle - object
        Sink - object
        Cloth - object
        Pen - object
        Chair - object
        Boots - object
        StoveKnob - object
        Box - object
        Egg - object
        TeddyBear - object
        Pencil - object
        ShowerDoor - object
        SprayBottle - object
        Book - object
        Apple - object
        Ladle - object
        PaperTowel - object
        Watch - object
        Laptop - object
        Pot - object
        HandTowel - object
        Plate - object
        Curtains - object
        FloorLamp - object
        PaperTowelRoll - object
        BasketBall - object
        Painting - object
        LightSwitch - object
        RemoteControl - object
        TissueBox - object
        DishSponge - object
        CreditCard - object
        Mirror - object
        ToiletPaperRoll - object
        Bowl - object
        ToiletPaper - object
        LaundryHamperLid - object
        Spoon - object
        Window - object
        SoapBar - object
        Lettuce - object
        Blinds - object
        SpatulaType - otype
        KeyChainType - otype
        ForkType - otype
        TomatoType - otype
        CDType - otype
        VaseType - otype
        ButterKnifeType - otype
        ScrubBrushType - otype
        TowelType - otype
        TennisRacketType - otype
        ShowerGlassType - otype
        BreadType - otype
        CupType - otype
        FootstoolType - otype
        NewspaperType - otype
        SoapBottleType - otype
        BathtubType - otype
        KnifeType - otype
        PepperShakerType - otype
        StatueType - otype
        MugType - otype
        GlassbottleType - otype
        AlarmClockType - otype
        BaseballBatType - otype
        HousePlantType - otype
        WineBottleType - otype
        DeskLampType - otype
        TelevisionType - otype
        PotatoType - otype
        WateringCanType - otype
        PosterType - otype
        SaltShakerType - otype
        PanType - otype
        PillowType - otype
        PlungerType - otype
        CellPhoneType - otype
        CandleType - otype
        KettleType - otype
        SinkType - otype
        ClothType - otype
        PenType - otype
        ChairType - otype
        BootsType - otype
        StoveKnobType - otype
        BoxType - otype
        EggType - otype
        TeddyBearType - otype
        PencilType - otype
        ShowerDoorType - otype
        SprayBottleType - otype
        BookType - otype
        AppleType - otype
        LadleType - otype
        PaperTowelType - otype
        WatchType - otype
        LaptopType - otype
        PotType - otype
        HandTowelType - otype
        PlateType - otype
        CurtainsType - otype
        FloorLampType - otype
        PaperTowelRollType - otype
        BasketBallType - otype
        PaintingType - otype
        LightSwitchType - otype
        RemoteControlType - otype
        TissueBoxType - otype
        DishSpongeType - otype
        CreditCardType - otype
        MirrorType - otype
        ToiletPaperRollType - otype
        BowlType - otype
        ToiletPaperType - otype
        LaundryHamperLidType - otype
        SpoonType - otype
        WindowType - otype
        SoapBarType - otype
        LettuceType - otype
        BlindsType - otype
        ToasterType - rtype
        CoffeeMachineType - rtype
        ToiletType - rtype
        ToiletPaperHangerType - rtype
        FridgeType - rtype
        DeskType - rtype
        SinkBasinType - rtype
        CartType - rtype
        CoffeeTableType - rtype
        DresserType - rtype
        PaintingHangerType - rtype
        DrawerType - rtype
        HandTowelHolderType - rtype
        MicrowaveType - rtype
        TVStandType - rtype
        OttomanType - rtype
        BathtubBasinType - rtype
        SideTableType - rtype
        SafeType - rtype
        BedType - rtype
        CabinetType - rtype
        StoveBurnerType - rtype
        LaundryHamperType - rtype
        ShelfType - rtype
        DiningTableType - rtype
        SofaType - rtype
        GarbageCanType - rtype
        TowelHolderType - rtype
        ArmChairType - rtype
        CounterTopType - rtype


        Bowl_bar__plus_02_dot_02_bar__plus_00_dot_34_bar__minus_00_dot_28 - object
        Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01 - object
        Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01 - object
        Pan_bar__plus_01_dot_93_bar__plus_00_dot_07_bar__plus_00_dot_64 - object
        Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41 - object
        Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 - object
        Pot_bar__plus_01_dot_87_bar__plus_00_dot_94_bar__plus_01_dot_50 - object
        Cabinet_bar__plus_00_dot_13_bar__plus_00_dot_39_bar__plus_01_dot_77 - receptacle
        Cabinet_bar__plus_00_dot_35_bar__plus_00_dot_39_bar__plus_02_dot_36 - receptacle
        Cabinet_bar__plus_01_dot_51_bar__plus_00_dot_39_bar__plus_02_dot_36 - receptacle
        Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_00_dot_87 - receptacle
        Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_02_dot_35 - receptacle
        Cabinet_bar__plus_01_dot_97_bar__plus_02_dot_11_bar__plus_02_dot_62 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_28 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_87 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_00_dot_89 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_77 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_81 - receptacle
        Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_02_dot_62 - receptacle
        CoffeeMachine_bar__plus_02_dot_08_bar__plus_00_dot_93_bar__plus_02_dot_67 - receptacle
        CounterTop_bar__plus_01_dot_07_bar__plus_00_dot_97_bar__plus_02_dot_67 - receptacle
        CounterTop_bar__plus_02_dot_06_bar__plus_00_dot_97_bar__plus_00_dot_58 - receptacle
        DiningTable_bar__minus_00_dot_15_bar_00_dot_00_bar__plus_01_dot_07 - receptacle
        Drawer_bar__plus_01_dot_91_bar__plus_00_dot_77_bar__plus_02_dot_06 - receptacle
        Drawer_bar__plus_02_dot_17_bar__plus_00_dot_77_bar__plus_00_dot_58 - receptacle
        Drawer_bar__minus_00_dot_02_bar__plus_00_dot_77_bar__plus_02_dot_06 - receptacle
        Fridge_bar__plus_02_dot_10_bar__plus_00_dot_00_bar__minus_00_dot_28 - receptacle
        GarbageCan_bar__minus_00_dot_31_bar_00_dot_00_bar__minus_00_dot_81 - receptacle
        Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08 - receptacle
        Sink_bar__plus_00_dot_94_bar__plus_00_dot_94_bar__plus_02_dot_65_bar_SinkBasin - receptacle
        StoveBurner_bar__plus_01_dot_86_bar__plus_00_dot_93_bar__plus_01_dot_11 - receptacle
        StoveBurner_bar__plus_01_dot_87_bar__plus_00_dot_93_bar__plus_01_dot_50 - receptacle
        StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_11 - receptacle
        StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_50 - receptacle
        Toaster_bar__plus_02_dot_13_bar__plus_00_dot_93_bar__plus_00_dot_57 - receptacle
        loc_bar_5_bar_1_bar_1_bar__minus_15 - location
        loc_bar_6_bar_2_bar_1_bar_45 - location
        loc_bar_6_bar_0_bar_1_bar_60 - location
        loc_bar_4_bar_3_bar_1_bar_30 - location
        loc_bar_6_bar_8_bar_0_bar_45 - location
        loc_bar_6_bar__minus_1_bar_1_bar_60 - location
        loc_bar_2_bar_2_bar_3_bar_45 - location
        loc_bar_6_bar_8_bar_1_bar__minus_30 - location
        loc_bar_6_bar_8_bar_0_bar__minus_30 - location
        loc_bar_4_bar_6_bar_0_bar_60 - location
        loc_bar_5_bar_6_bar_1_bar_45 - location
        loc_bar_4_bar_8_bar_1_bar_45 - location
        loc_bar_6_bar_0_bar_1_bar_15 - location
        loc_bar_5_bar_2_bar_1_bar__minus_15 - location
        loc_bar_3_bar_8_bar_3_bar_45 - location
        loc_bar_4_bar_8_bar_0_bar_60 - location
        loc_bar_3_bar_6_bar_1_bar__minus_15 - location
        loc_bar_2_bar_4_bar_3_bar_60 - location
        loc_bar_3_bar_6_bar_0_bar_60 - location
        loc_bar_5_bar_7_bar_0_bar_30 - location
        loc_bar_2_bar_8_bar_3_bar_45 - location
        loc_bar_5_bar__minus_1_bar_1_bar_60 - location
        loc_bar_3_bar_4_bar_1_bar_45 - location
        loc_bar_3_bar_5_bar_1_bar__minus_15 - location
        loc_bar_5_bar_2_bar_1_bar_45 - location
        loc_bar_6_bar_2_bar_1_bar_60 - location
        loc_bar_4_bar_7_bar_3_bar_60 - location
        loc_bar_3_bar_6_bar_1_bar_45 - location
        loc_bar_1_bar__minus_3_bar_3_bar_60 - location
        loc_bar_5_bar_4_bar_1_bar_45 - location
        loc_bar_5_bar_4_bar_3_bar_30 - location
        )


    (:init
        (= (totalCost) 0)


        (receptacleType Drawer_bar__minus_00_dot_02_bar__plus_00_dot_77_bar__plus_02_dot_06 DrawerType)
        (receptacleType Drawer_bar__plus_02_dot_17_bar__plus_00_dot_77_bar__plus_00_dot_58 DrawerType)
        (receptacleType StoveBurner_bar__plus_01_dot_86_bar__plus_00_dot_93_bar__plus_01_dot_11 StoveBurnerType)
        (receptacleType Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_87 CabinetType)
        (receptacleType GarbageCan_bar__minus_00_dot_31_bar_00_dot_00_bar__minus_00_dot_81 GarbageCanType)
        (receptacleType Toaster_bar__plus_02_dot_13_bar__plus_00_dot_93_bar__plus_00_dot_57 ToasterType)
        (receptacleType Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_77 CabinetType)
        (receptacleType CounterTop_bar__plus_01_dot_07_bar__plus_00_dot_97_bar__plus_02_dot_67 CounterTopType)
        (receptacleType Drawer_bar__plus_01_dot_91_bar__plus_00_dot_77_bar__plus_02_dot_06 DrawerType)
        (receptacleType Fridge_bar__plus_02_dot_10_bar__plus_00_dot_00_bar__minus_00_dot_28 FridgeType)
        (receptacleType Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_00_dot_87 CabinetType)
        (receptacleType Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_02_dot_35 CabinetType)
        (receptacleType CounterTop_bar__plus_02_dot_06_bar__plus_00_dot_97_bar__plus_00_dot_58 CounterTopType)
        (receptacleType Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_00_dot_89 CabinetType)
        (receptacleType StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_50 StoveBurnerType)
        (receptacleType Cabinet_bar__plus_00_dot_13_bar__plus_00_dot_39_bar__plus_01_dot_77 CabinetType)
        (receptacleType Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08 MicrowaveType)
        (receptacleType StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_11 StoveBurnerType)
        (receptacleType Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_28 CabinetType)
        (receptacleType Cabinet_bar__plus_01_dot_51_bar__plus_00_dot_39_bar__plus_02_dot_36 CabinetType)
        (receptacleType DiningTable_bar__minus_00_dot_15_bar_00_dot_00_bar__plus_01_dot_07 DiningTableType)
        (receptacleType Cabinet_bar__plus_00_dot_35_bar__plus_00_dot_39_bar__plus_02_dot_36 CabinetType)
        (receptacleType Sink_bar__plus_00_dot_94_bar__plus_00_dot_94_bar__plus_02_dot_65_bar_SinkBasin SinkBasinType)
        (receptacleType CoffeeMachine_bar__plus_02_dot_08_bar__plus_00_dot_93_bar__plus_02_dot_67 CoffeeMachineType)
        (receptacleType StoveBurner_bar__plus_01_dot_87_bar__plus_00_dot_93_bar__plus_01_dot_50 StoveBurnerType)
        (objectType Pan_bar__plus_01_dot_93_bar__plus_00_dot_07_bar__plus_00_dot_64 PanType)
        (objectType Bowl_bar__plus_02_dot_02_bar__plus_00_dot_34_bar__minus_00_dot_28 BowlType)
        (objectType Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41 PlateType)
        (objectType Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01 MugType)
        (objectType Pot_bar__plus_01_dot_87_bar__plus_00_dot_94_bar__plus_01_dot_50 PotType)
        (objectType Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 PotatoType)
        (objectType Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01 CupType)
        (isReceptacleObject Pan_bar__plus_01_dot_93_bar__plus_00_dot_07_bar__plus_00_dot_64)
        (isReceptacleObject Bowl_bar__plus_02_dot_02_bar__plus_00_dot_34_bar__minus_00_dot_28)
        (isReceptacleObject Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41)
        (isReceptacleObject Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01)
        (isReceptacleObject Pot_bar__plus_01_dot_87_bar__plus_00_dot_94_bar__plus_01_dot_50)
        (isReceptacleObject Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01)
        (openable Drawer_bar__minus_00_dot_02_bar__plus_00_dot_77_bar__plus_02_dot_06)
        (openable Drawer_bar__plus_02_dot_17_bar__plus_00_dot_77_bar__plus_00_dot_58)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_87)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_77)
        (openable Drawer_bar__plus_01_dot_91_bar__plus_00_dot_77_bar__plus_02_dot_06)
        (openable Fridge_bar__plus_02_dot_10_bar__plus_00_dot_00_bar__minus_00_dot_28)
        (openable Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_00_dot_87)
        (openable Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_02_dot_35)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_00_dot_89)
        (openable Cabinet_bar__plus_00_dot_13_bar__plus_00_dot_39_bar__plus_01_dot_77)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_81)
        (openable Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08)
        (openable Cabinet_bar__plus_01_dot_97_bar__plus_02_dot_11_bar__plus_02_dot_62)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_28)
        (openable Cabinet_bar__plus_01_dot_51_bar__plus_00_dot_39_bar__plus_02_dot_36)
        (openable Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_02_dot_62)
        (openable Cabinet_bar__plus_00_dot_35_bar__plus_00_dot_39_bar__plus_02_dot_36)
        
        (atLocation agent1 loc_bar_5_bar_4_bar_3_bar_30)
        
        (cleanable Pan_bar__plus_01_dot_93_bar__plus_00_dot_07_bar__plus_00_dot_64)
        (cleanable Bowl_bar__plus_02_dot_02_bar__plus_00_dot_34_bar__minus_00_dot_28)
        (cleanable Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41)
        (cleanable Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01)
        (cleanable Pot_bar__plus_01_dot_87_bar__plus_00_dot_94_bar__plus_01_dot_50)
        (cleanable Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08)
        (cleanable Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01)
        
        (heatable Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41)
        (heatable Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01)
        (heatable Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08)
        (heatable Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01)
        (coolable Pan_bar__plus_01_dot_93_bar__plus_00_dot_07_bar__plus_00_dot_64)
        (coolable Bowl_bar__plus_02_dot_02_bar__plus_00_dot_34_bar__minus_00_dot_28)
        (coolable Plate_bar__minus_00_dot_31_bar__plus_00_dot_78_bar__plus_00_dot_41)
        (coolable Mug_bar__plus_02_dot_02_bar__plus_01_dot_51_bar__minus_00_dot_01)
        (coolable Pot_bar__plus_01_dot_87_bar__plus_00_dot_94_bar__plus_01_dot_50)
        (coolable Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08)
        (coolable Cup_bar__plus_02_dot_25_bar__plus_00_dot_60_bar__minus_00_dot_01)
        
        
        
        
        
        
        
        (inReceptacle Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 CounterTop_bar__plus_01_dot_07_bar__plus_00_dot_97_bar__plus_02_dot_67)
        (inReceptacle Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08)
        (wasInReceptacle  Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 CounterTop_bar__plus_01_dot_07_bar__plus_00_dot_97_bar__plus_02_dot_67)
        (wasInReceptacle  Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_45) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_0_bar_60) 17)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_60) 10)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_1_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_2_bar_4_bar_3_bar_60) 14)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_4_bar_3_bar_1_bar_30) 11)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_1_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar_45) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_0_bar_60) 16)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_7_bar_0_bar_30) 11)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_1_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_2_bar_8_bar_3_bar_45) 17)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_4_bar_1_bar_45) 14)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_1_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_1_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_1_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_4_bar_7_bar_3_bar_60) 15)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar_45) 16)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_1_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_4_bar_6_bar_0_bar_60) 15)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_6_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_1_bar_45) 17)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_1_bar__minus_3_bar_3_bar_60) 16)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_1_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_15) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_1_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_1_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_3_bar_30) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_1_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_5_bar_1_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_1_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 12)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 13)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 6)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_45) 16)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 9)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_2_bar_1_bar_45) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 8)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 12)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 11)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 7)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_45) 15)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 7)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_2_bar_1_bar_45) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 2)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 2)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 11)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 16)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 16)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 13)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 14)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_45) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 8)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_2_bar_1_bar_45) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_45) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 7)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_6_bar_2_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 6)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 6)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 15)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 15)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_3_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 9)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_8_bar_3_bar_45) 10)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 7)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 7)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 7)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 2)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 2)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 16)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 15)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 7)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_3_bar_8_bar_3_bar_45) 10)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 11)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 10)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 11)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_8_bar_3_bar_45) 11)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 13)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 8)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 9)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_8_bar_3_bar_45) 12)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 7)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 4)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 4)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 17)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_8_bar_3_bar_45) 17)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_8_bar_3_bar_45) 17)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 12)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_8_bar_3_bar_45) 12)
        (= (distance loc_bar_3_bar_8_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_8_bar_3_bar_45) 15)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 14)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 11)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_4_bar_8_bar_0_bar_60) 11)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 6)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 6)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 9)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_4_bar_8_bar_0_bar_60) 9)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 5)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 5)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 15)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 14)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_0_bar_60) 13)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 13)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 10)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_4_bar_8_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 5)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 5)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 13)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 13)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_4_bar_8_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 7)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 3)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 3)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 18)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_4_bar_8_bar_0_bar_60) 18)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_4_bar_8_bar_0_bar_60) 17)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_4_bar_8_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 11)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_4_bar_8_bar_0_bar_60) 11)
        (= (distance loc_bar_4_bar_8_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_0_bar_60) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 19)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_60) 19)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 11)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 12)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_0_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 13)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 14)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_0_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 16)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 4)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 4)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 7)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 7)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 13)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 18)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_60) 18)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 9)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 10)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 17)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_0_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 5)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 5)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 12)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 20)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 20)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 17)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_0_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 13)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 16)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 16)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 11)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 4)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_0_bar_1_bar_60) 4)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_0_bar_1_bar_60) 10)
        (= (distance loc_bar_6_bar_0_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_60) 14)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_2_bar_4_bar_3_bar_60) 11)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_6_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 7)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_6_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_2_bar_8_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar__minus_1_bar_1_bar_60) 18)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 18)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar__minus_1_bar_1_bar_60) 17)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_3_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_3_bar_5_bar_1_bar__minus_15) 4)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar__minus_15) 4)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_2_bar_2_bar_3_bar_45) 12)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_1_bar__minus_30) 9)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_6_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_60) 15)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_4_bar_7_bar_3_bar_60) 10)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar_45) 5)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 5)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_6_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_4_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 7)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_1_bar_45) 10)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_1_bar__minus_3_bar_3_bar_60) 19)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar__minus_15) 19)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_6_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_6_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 8)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_2_bar_4_bar_3_bar_60) 8)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 13)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 13)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 7)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 12)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_2_bar_4_bar_3_bar_60) 12)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 8)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 12)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 11)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 5)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 5)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 9)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 6)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 17)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_2_bar_4_bar_3_bar_60) 17)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 9)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 10)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 7)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 19)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_2_bar_4_bar_3_bar_60) 19)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 8)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 9)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 10)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 11)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_2_bar_4_bar_3_bar_60) 11)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_2_bar_4_bar_3_bar_60) 14)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_2_bar_4_bar_3_bar_60) 7)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_2_bar_4_bar_3_bar_60) 10)
        (= (distance loc_bar_2_bar_4_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_2_bar_4_bar_3_bar_60) 14)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_8_bar_0_bar_45) 10)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_7_bar_0_bar_30) 7)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_4_bar_3_bar_1_bar_30) 7)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_2_bar_8_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 11)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 11)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 8)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_4_bar_3_bar_1_bar_30) 11)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 6)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_2_bar_2_bar_3_bar_45) 7)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 7)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_8_bar_1_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_4_bar_3_bar_1_bar_30) 14)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_2_bar_1_bar_60) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 8)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_4_bar_7_bar_3_bar_60) 9)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 9)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_3_bar_6_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 10)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_4_bar_3_bar_1_bar_30) 14)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_4_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 7)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 8)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_4_bar_8_bar_1_bar_45) 9)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 9)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_1_bar__minus_3_bar_3_bar_60) 14)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_4_bar_3_bar_1_bar_30) 14)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_6_bar_0_bar_1_bar_15) 9)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_4_bar_3_bar_1_bar_30) 9)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_4_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_4_bar_3_bar_1_bar_30) 6)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_4_bar_3_bar_30) 5)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_4_bar_3_bar_1_bar_30) 5)
        (= (distance loc_bar_4_bar_3_bar_1_bar_30 loc_bar_5_bar_2_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_4_bar_3_bar_1_bar_30) 9)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_8_bar_0_bar_45) 8)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 6)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 15)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 11)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar_45) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 14)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 7)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_8_bar_0_bar_45) 7)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 8)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 8)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 9)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 9)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 6)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_8_bar_0_bar_45) 6)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 10)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 7)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 6)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 6)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 21)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar_45) 21)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_8_bar_0_bar_45) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar_45) 9)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_8_bar_0_bar_45) 10)
        (= (distance loc_bar_6_bar_8_bar_0_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar_45) 15)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 6)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 14)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 13)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 13)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 7)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_6_bar_0_bar_60) 13)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 6)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 6)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 3)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 3)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_6_bar_0_bar_60) 14)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 4)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 4)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 5)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 5)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 6)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 6)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 15)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_6_bar_0_bar_60) 15)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 16)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_6_bar_0_bar_60) 16)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_0_bar_60) 9)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_3_bar_6_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_0_bar_60) 16)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_2_bar_8_bar_3_bar_45) 7)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 7)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 15)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 14)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 10)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_7_bar_0_bar_30) 11)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 10)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_2_bar_2_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 13)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_8_bar_1_bar__minus_30) 8)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_2_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 12)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_4_bar_7_bar_3_bar_60) 5)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 5)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_3_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_8_bar_0_bar__minus_30) 9)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_7_bar_0_bar_30) 9)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_4_bar_6_bar_0_bar_60) 9)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 9)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_6_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 6)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_4_bar_8_bar_1_bar_45) 7)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 7)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_1_bar__minus_3_bar_3_bar_60) 20)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_7_bar_0_bar_30) 20)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_6_bar_0_bar_1_bar_15) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_7_bar_0_bar_30) 13)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_7_bar_0_bar_30) 8)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_4_bar_3_bar_30) 7)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_7_bar_0_bar_30) 7)
        (= (distance loc_bar_5_bar_7_bar_0_bar_30 loc_bar_5_bar_2_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_7_bar_0_bar_30) 13)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 17)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 16)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_2_bar_8_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 12)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 9)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_2_bar_8_bar_3_bar_45) 12)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 14)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 9)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 6)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_2_bar_8_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 9)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 9)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 5)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 5)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 16)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_2_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_2_bar_8_bar_3_bar_45) 17)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_2_bar_8_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 13)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_2_bar_8_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_8_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_2_bar_8_bar_3_bar_45) 16)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 6)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 6)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 14)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 19)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar__minus_1_bar_1_bar_60) 19)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 18)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar__minus_1_bar_1_bar_60) 18)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 6)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 6)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 13)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 16)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 16)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 17)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar__minus_1_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 13)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 13)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 14)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 17)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 10)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 7)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar__minus_1_bar_1_bar_60) 7)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar__minus_1_bar_1_bar_60) 12)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 11)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 13)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 13)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 18)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar__minus_1_bar_1_bar_60) 18)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 7)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 19)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar__minus_1_bar_1_bar_60) 19)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 7)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 7)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 12)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 12)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 15)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 15)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 18)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar__minus_1_bar_1_bar_60) 18)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 12)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 12)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 11)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 16)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 16)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 9)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar__minus_1_bar_1_bar_60) 9)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 8)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar__minus_1_bar_1_bar_60) 8)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar__minus_1_bar_1_bar_60) 9)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 10)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar__minus_1_bar_1_bar_60) 10)
        (= (distance loc_bar_5_bar__minus_1_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar__minus_1_bar_1_bar_60) 12)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_3_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 6)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_4_bar_1_bar_45) 15)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 8)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 5)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 5)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_4_bar_1_bar_45) 14)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 6)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 6)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 8)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 13)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_4_bar_1_bar_45) 13)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_4_bar_1_bar_45) 12)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 3)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_4_bar_1_bar_45) 3)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 6)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_4_bar_1_bar_45) 6)
        (= (distance loc_bar_3_bar_4_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_4_bar_1_bar_45) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_1_bar__minus_30) 10)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_4_bar_7_bar_3_bar_60) 11)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar__minus_30) 9)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_5_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_4_bar_6_bar_0_bar_60) 9)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_6_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_1_bar_45) 11)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_1_bar__minus_3_bar_3_bar_60) 18)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_5_bar_1_bar__minus_15) 18)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_15) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_5_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_5_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_3_bar_30) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_5_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_3_bar_5_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_5_bar_1_bar__minus_15) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 6)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_2_bar_1_bar_45) 15)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 3)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 3)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 10)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 11)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_2_bar_1_bar_45) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 10)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 12)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 13)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar_45) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 8)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_2_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 5)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar_45) 5)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 6)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_2_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_2_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 5)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_2_bar_1_bar_45) 5)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 18)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_2_bar_2_bar_3_bar_45) 18)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 8)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 13)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 13)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 19)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_2_bar_2_bar_3_bar_45) 19)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 10)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 11)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 10)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_2_bar_2_bar_3_bar_45) 8)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 11)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_2_bar_2_bar_3_bar_45) 11)
        (= (distance loc_bar_2_bar_2_bar_3_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 10)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_2_bar_2_bar_3_bar_45) 10)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_2_bar_1_bar_60) 15)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_4_bar_7_bar_3_bar_60) 12)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_3_bar_6_bar_1_bar_45) 15)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_8_bar_0_bar__minus_30) 2)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_8_bar_1_bar__minus_30) 2)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_4_bar_6_bar_0_bar_60) 14)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_6_bar_1_bar_45) 13)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_4_bar_8_bar_1_bar_45) 12)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_1_bar__minus_3_bar_3_bar_60) 25)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_8_bar_1_bar__minus_30) 25)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_8_bar_1_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_4_bar_1_bar_45) 15)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_1_bar__minus_30) 15)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_4_bar_3_bar_30) 12)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_8_bar_1_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_1_bar__minus_30 loc_bar_5_bar_2_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_1_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 11)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 17)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 17)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 11)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 11)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 14)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 14)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_2_bar_1_bar_60) 14)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_2_bar_1_bar_60) 11)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_2_bar_1_bar_60) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_2_bar_1_bar_60) 9)
        (= (distance loc_bar_6_bar_2_bar_1_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 15)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_2_bar_1_bar_60) 15)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 6)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 6)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 13)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_4_bar_7_bar_3_bar_60) 13)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 5)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 5)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 6)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 5)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 5)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 16)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_4_bar_7_bar_3_bar_60) 16)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 15)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_4_bar_7_bar_3_bar_60) 15)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_4_bar_7_bar_3_bar_60) 8)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 11)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_4_bar_7_bar_3_bar_60) 11)
        (= (distance loc_bar_4_bar_7_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_4_bar_7_bar_3_bar_60) 14)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_3_bar_6_bar_1_bar_45) 12)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 4)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 4)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 3)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 3)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 10)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 10)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 15)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_3_bar_6_bar_1_bar_45) 15)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_3_bar_6_bar_1_bar_45) 14)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 7)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_3_bar_6_bar_1_bar_45) 7)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 8)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_3_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_3_bar_6_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 13)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_3_bar_6_bar_1_bar_45) 13)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_4_bar_6_bar_0_bar_60) 16)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 16)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_6_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 12)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_4_bar_8_bar_1_bar_45) 11)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 11)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_1_bar__minus_3_bar_3_bar_60) 27)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_8_bar_0_bar__minus_30) 27)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_6_bar_0_bar_1_bar_15) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_6_bar_8_bar_0_bar__minus_30) 17)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_4_bar_1_bar_45) 14)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_4_bar_3_bar_30) 14)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_8_bar_0_bar__minus_30) 14)
        (= (distance loc_bar_6_bar_8_bar_0_bar__minus_30 loc_bar_5_bar_2_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_8_bar_0_bar__minus_30) 16)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 4)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 4)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 5)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 5)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_1_bar__minus_3_bar_3_bar_60) 16)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_4_bar_6_bar_0_bar_60) 16)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 15)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_4_bar_6_bar_0_bar_60) 15)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_4_bar_6_bar_0_bar_60) 8)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_4_bar_6_bar_0_bar_60) 9)
        (= (distance loc_bar_4_bar_6_bar_0_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_4_bar_6_bar_0_bar_60) 14)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 8)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 17)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_6_bar_1_bar_45) 17)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 12)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_6_bar_1_bar_45) 12)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 5)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_6_bar_1_bar_45) 5)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 6)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_6_bar_1_bar_45) 6)
        (= (distance loc_bar_5_bar_6_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 11)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_6_bar_1_bar_45) 11)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 18)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_4_bar_8_bar_1_bar_45) 18)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 15)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_4_bar_8_bar_1_bar_45) 15)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_4_bar_1_bar_45) 8)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_4_bar_8_bar_1_bar_45) 8)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_4_bar_8_bar_1_bar_45) 9)
        (= (distance loc_bar_4_bar_8_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 14)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_4_bar_8_bar_1_bar_45) 14)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_6_bar_0_bar_1_bar_15) 14)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_1_bar__minus_3_bar_3_bar_60) 14)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_4_bar_1_bar_45) 15)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_1_bar__minus_3_bar_3_bar_60) 15)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_4_bar_3_bar_30) 18)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_1_bar__minus_3_bar_3_bar_60) 18)
        (= (distance loc_bar_1_bar__minus_3_bar_3_bar_60 loc_bar_5_bar_2_bar_1_bar__minus_15) 17)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_1_bar__minus_3_bar_3_bar_60) 17)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_4_bar_1_bar_45) 13)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_6_bar_0_bar_1_bar_15) 13)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_4_bar_3_bar_30) 11)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_6_bar_0_bar_1_bar_15) 11)
        (= (distance loc_bar_6_bar_0_bar_1_bar_15 loc_bar_5_bar_2_bar_1_bar__minus_15) 16)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_6_bar_0_bar_1_bar_15) 16)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_4_bar_3_bar_30) 4)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_4_bar_1_bar_45) 4)
        (= (distance loc_bar_5_bar_4_bar_1_bar_45 loc_bar_5_bar_2_bar_1_bar__minus_15) 9)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_1_bar_45) 9)
        (= (distance loc_bar_5_bar_4_bar_3_bar_30 loc_bar_5_bar_2_bar_1_bar__minus_15) 8)
        (= (distance loc_bar_5_bar_2_bar_1_bar__minus_15 loc_bar_5_bar_4_bar_3_bar_30) 8)
        (receptacleAtLocation Cabinet_bar__plus_00_dot_13_bar__plus_00_dot_39_bar__plus_01_dot_77 loc_bar_4_bar_7_bar_3_bar_60)
        (receptacleAtLocation Cabinet_bar__plus_00_dot_35_bar__plus_00_dot_39_bar__plus_02_dot_36 loc_bar_4_bar_6_bar_0_bar_60)
        (receptacleAtLocation Cabinet_bar__plus_01_dot_51_bar__plus_00_dot_39_bar__plus_02_dot_36 loc_bar_3_bar_6_bar_0_bar_60)
        (receptacleAtLocation Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_00_dot_87 loc_bar_3_bar_4_bar_1_bar_45)
        (receptacleAtLocation Cabinet_bar__plus_01_dot_76_bar__plus_00_dot_39_bar__plus_02_dot_35 loc_bar_3_bar_6_bar_1_bar_45)
        (receptacleAtLocation Cabinet_bar__plus_01_dot_97_bar__plus_02_dot_11_bar__plus_02_dot_62 loc_bar_6_bar_8_bar_0_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_28 loc_bar_5_bar_1_bar_1_bar__minus_15)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_01_dot_81_bar__plus_00_dot_87 loc_bar_5_bar_2_bar_1_bar__minus_15)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_00_dot_89 loc_bar_3_bar_6_bar_1_bar__minus_15)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_77 loc_bar_3_bar_5_bar_1_bar__minus_15)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_01_dot_81 loc_bar_6_bar_8_bar_1_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__plus_02_dot_04_bar__plus_02_dot_11_bar__plus_02_dot_62 loc_bar_6_bar_8_bar_0_bar__minus_30)
        (receptacleAtLocation CoffeeMachine_bar__plus_02_dot_08_bar__plus_00_dot_93_bar__plus_02_dot_67 loc_bar_6_bar_8_bar_0_bar_45)
        (receptacleAtLocation CounterTop_bar__plus_01_dot_07_bar__plus_00_dot_97_bar__plus_02_dot_67 loc_bar_5_bar_7_bar_0_bar_30)
        (receptacleAtLocation CounterTop_bar__plus_02_dot_06_bar__plus_00_dot_97_bar__plus_00_dot_58 loc_bar_5_bar_2_bar_1_bar_45)
        (receptacleAtLocation DiningTable_bar__minus_00_dot_15_bar_00_dot_00_bar__plus_01_dot_07 loc_bar_2_bar_4_bar_3_bar_60)
        (receptacleAtLocation Drawer_bar__plus_01_dot_91_bar__plus_00_dot_77_bar__plus_02_dot_06 loc_bar_4_bar_8_bar_1_bar_45)
        (receptacleAtLocation Drawer_bar__plus_02_dot_17_bar__plus_00_dot_77_bar__plus_00_dot_58 loc_bar_4_bar_3_bar_1_bar_30)
        (receptacleAtLocation Drawer_bar__minus_00_dot_02_bar__plus_00_dot_77_bar__plus_02_dot_06 loc_bar_3_bar_8_bar_3_bar_45)
        (receptacleAtLocation Fridge_bar__plus_02_dot_10_bar__plus_00_dot_00_bar__minus_00_dot_28 loc_bar_5_bar__minus_1_bar_1_bar_60)
        (receptacleAtLocation GarbageCan_bar__minus_00_dot_31_bar_00_dot_00_bar__minus_00_dot_81 loc_bar_1_bar__minus_3_bar_3_bar_60)
        (receptacleAtLocation Microwave_bar__minus_00_dot_31_bar__plus_00_dot_93_bar__plus_02_dot_08 loc_bar_2_bar_8_bar_3_bar_45)
        (receptacleAtLocation Sink_bar__plus_00_dot_94_bar__plus_00_dot_94_bar__plus_02_dot_65_bar_SinkBasin loc_bar_4_bar_8_bar_0_bar_60)
        (receptacleAtLocation StoveBurner_bar__plus_01_dot_86_bar__plus_00_dot_93_bar__plus_01_dot_11 loc_bar_5_bar_4_bar_1_bar_45)
        (receptacleAtLocation StoveBurner_bar__plus_01_dot_87_bar__plus_00_dot_93_bar__plus_01_dot_50 loc_bar_5_bar_6_bar_1_bar_45)
        (receptacleAtLocation StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_11 loc_bar_5_bar_4_bar_1_bar_45)
        (receptacleAtLocation StoveBurner_bar__plus_02_dot_14_bar__plus_00_dot_93_bar__plus_01_dot_50 loc_bar_5_bar_6_bar_1_bar_45)
        (receptacleAtLocation Toaster_bar__plus_02_dot_13_bar__plus_00_dot_93_bar__plus_00_dot_57 loc_bar_6_bar_2_bar_1_bar_45)
        (objectAtLocation Potato_bar__minus_00_dot_31_bar__plus_01_dot_03_bar__plus_02_dot_08 loc_bar_2_bar_8_bar_3_bar_45)
        )


        (:goal
            (and
                (exists (?r - receptacle)
                    (exists (?o - object)
                        (and 
                            (coolable ?o)
                            (objectType ?o PotatoType) 
                            (receptacleType ?r DiningTableType)
                            (isCool ?o)
                            (inReceptacle ?o ?r) 
                        )
                    )
                )
                (forall (?re - receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    