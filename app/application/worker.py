import yaml
from wredis.streams import RedisStreamManager
from application.executor import train_run
from loguru import logger

def main():
    # Load Redis configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    redis_config = config.get("redis", {})
    
    stream_manager = RedisStreamManager(
        host=redis_config.get("REDIS_HOST"),
        port=redis_config.get("REDIS_PORT"),
        db=redis_config.get("REDIS_DB"),
        verbose=True
    )
    
    topic = redis_config.get("TOPIC", "training_queue")
    group_name = "training_group"
    consumer_name = "consumer-1"

    stream_manager.create_group(topic, group_name)

    logger.info(f"Worker listening on topic: {topic}, group: {group_name}")

    while True:
        try:
            # Read messages from the stream
            messages = stream_manager.read_stream(
                key=topic,
                group_name=group_name,
                consumer_name=consumer_name,
                count=1,
                block=0
            )

            if messages:
                for stream, message_list in messages:
                    for message_id, message_data in message_list:
                        logger.info(f"Processing message {message_id}: {message_data}")
                        
                        # The message should contain the config_path
                        config_path = message_data.get("config_path")
                        trial_number = message_data.get("trial_number", 1)

                        if config_path:
                            try:
                                train_run(config_path=config_path, trial_number=trial_number)
                                # Acknowledge the message
                                stream_manager.ack_message(topic, group_name, message_id)
                                logger.info(f"Message {message_id} processed and acknowledged.")
                            except Exception as e:
                                logger.error(f"Error processing message {message_id}: {e}")
                        else:
                            logger.warning(f"Message {message_id} does not contain 'config_path'")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
